"""Train the encoder-only Transformer on IMDb under several configurations.

Examples
--------
    python train_imdb.py                          # the five configurations
    python train_imdb.py --experiments base       # a single one
    python train_imdb.py --experiments base --init xavier --seed 0 --out-dir runs/xavier_s0
"""

import argparse
import json
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm # progress bar

from model import TransformerEncoderClassifier


# ---------------------------------------------------------------------------
# Dataset IMDb + Tokenizer
# ---------------------------------------------------------------------------


class ImdbDataset(Dataset):
    """
    Wraps the HuggingFace split and builds:
        - encoder_input: tensor (seq_len,)
        - encoder_mask: tensor (1, 1, seq_len) with 1 on valid tokens, 0 on PAD
        - label: 0 (negative) or 1 (positive)
    """

    def __init__(self, hf_ds, tokenizer: Tokenizer, seq_len: int):
        super().__init__()
        self.ds = hf_ds # HuggingFace dataset split, text + label
        self.tokenizer = tokenizer 
        self.seq_len = seq_len

        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.sos_id = tokenizer.token_to_id("[SOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx): 
        item = self.ds[idx]
        text = item["text"]
        label = int(item["label"])  # 0 = neg, 1 = pos

        # Tokenization
        ids = self.tokenizer.encode(text).ids # list of token IDs
        
        max_tokens = self.seq_len - 2 # seq_len has included SOS and EOS

        # We truncate the text if it is too long
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]

        # [SOS] ... tokens ... [EOS] + PADs
        encoder_input = [self.sos_id] + ids + [self.eos_id]
        pad_len = self.seq_len - len(encoder_input)
        if pad_len > 0:
            encoder_input += [self.pad_id] * pad_len

        encoder_input = torch.tensor(encoder_input, dtype=torch.long)

        # mask: 1 where there is a real token, 0 where there is PAD
        encoder_mask = (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0).int()
        # final shape in batch: (B, 1, 1, seq_len)

        return {
            "encoder_input": encoder_input,
            "encoder_mask": encoder_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


def imdb_text_iterator(hf_split):
    """Simple iterator over texts to train the tokenizer."""
    for item in hf_split:
        yield item["text"] # yield is used to not iterate all at once


def build_tokenizer_imdb(tokenizer_path: str, ds_iter):
    """
    Loads the tokenizer if it exists; otherwise, trains it from IMDb (train)
    and saves it to tokenizer_path.
    """
    tokenizer_file = Path(tokenizer_path)
    if tokenizer_file.exists():
        print(f"Loading tokenizer from {tokenizer_file}")
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
    else:
        print("Building tokenizer from IMDb dataset...")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(ds_iter, trainer=trainer)
        tokenizer.save(str(tokenizer_file))
        print(f"Tokenizer saved to {tokenizer_file}")
    return tokenizer


def create_dataloaders(train_raw, tokenizer, seq_len: int, batch_size: int):
    """
    Builds Dataset + DataLoaders for train / val with the same split
    (random but fixed with a seed in order to compare experiments).
    """
    full_train = ImdbDataset(train_raw, tokenizer, seq_len)

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size

    # Fixed seed so that the split is the same in all experiments
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Training a single experiment
# ---------------------------------------------------------------------------


def train_one_experiment(
    name: str,
    device: torch.device,
    train_raw,
    tokenizer: Tokenizer,
    config: dict,
    init: str = "default",
    checkpoint_dir: Path = Path("checkpoints"),
):
    """
    Runs an experiment with a hyperparameter configuration and
    returns per-epoch metrics + timings.
    """
    print("=" * 80)
    print(f"Experiment: {name}")
    print(f"Config: {config}")
    print("=" * 80)

    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    d_ff = config["d_ff"]
    dropout = config["dropout"]
    num_epochs = config["num_epochs"]
    lr = config.get("lr", 1e-4)

    train_dl, val_dl = create_dataloaders(train_raw, tokenizer, seq_len, batch_size)

    vocab_size = tokenizer.get_vocab_size()
    print("Vocab size:", vocab_size)

    model = TransformerEncoderClassifier(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_classes=2,  # negative / positive
        d_model=d_model,
        N=num_layers,
        h=num_heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    if INITS[init] is not None:
        init_xavier(model, INITS[init])

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "experiment": name,
        "config": config,
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_time": [],
        "total_time": None,
    }

    exp_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        
        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            enc_in = batch["encoder_input"].to(device)      # (B, seq_len)
            enc_mask = batch["encoder_mask"].to(device)     # (B, 1, 1, seq_len)
            labels = batch["label"].to(device)              # (B,)

            logits = model(enc_in, enc_mask)                # (B, 2)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * enc_in.size(0)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += enc_in.size(0)

            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct / total):.4f}",
            })

        avg_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dl:
                enc_in = batch["encoder_input"].to(device)
                enc_mask = batch["encoder_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(enc_in, enc_mask)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += enc_in.size(0)

        val_acc = val_correct / val_total

        epoch_time = time.time() - epoch_start

        print(
            f"[{name}] Epoch {epoch+1}/{num_epochs} "
            f"| Train loss: {avg_loss:.4f} "
            f"| Train acc: {train_acc:.4f} "
            f"| Val acc: {val_acc:.4f} "
            f"| Epoch time: {epoch_time:.2f} s"
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

    total_time = time.time() - exp_start_time
    history["total_time"] = total_time
    print(f"[{name}] Total training time: {total_time:.2f} s")

     # Save model weights for this experiment
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / f"encoder_imdb_{name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model for experiment '{name}' saved to {model_path}")


    return history


# --init value -> which matrices get Xavier-uniform (None keeps PyTorch's defaults)
INITS = {
    "default": None,
    "xavier": "all",
    "xavier_embedding": "embedding",
    "xavier_layers": "layers",
}


def init_xavier(model: nn.Module, part: str = "all") -> None:
    """Xavier-uniform initialisation, which the upstream code applies to every matrix.

    part: "all", only the token "embedding", or all the other "layers".
    """
    if part not in ("all", "embedding", "layers"):
        raise ValueError(f"unknown part: {part}")
    for name, p in model.named_parameters():
        is_embedding = name.startswith("embed.")
        if p.dim() > 1 and (part == "all" or is_embedding == (part == "embedding")):
            nn.init.xavier_uniform_(p)


# ---------------------------------------------------------------------------
# Utilities to save results and plot
# ---------------------------------------------------------------------------


def save_results_json(all_histories, path="results_imdb_transformer.json"):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2)
    print(f"Results saved to {path}")



def save_results_csv(all_histories, path="results_imdb_transformer.csv"):
    path = Path(path)
    # We build a "long" table: one row per (experiment, epoch)
    rows = []
    for hist in all_histories:
        name = hist["experiment"]
        cfg = hist["config"]
        total_time = hist.get("total_time", None)
        for i, epoch in enumerate(hist["epoch"]):
            rows.append({
                "experiment": name,
                "epoch": epoch,
                "train_loss": hist["train_loss"][i],
                "train_acc": hist["train_acc"][i],
                "val_acc": hist["val_acc"][i],
                "epoch_time": hist["epoch_time"][i],
                "total_time": total_time,
                "seq_len": cfg["seq_len"],
                "batch_size": cfg["batch_size"],
                "d_model": cfg["d_model"],
                "num_layers": cfg["num_layers"],
                "num_heads": cfg["num_heads"],
                "d_ff": cfg["d_ff"],
                "dropout": cfg["dropout"],
                "num_epochs": cfg["num_epochs"],
                "lr": cfg.get("lr", 1e-4),
            })

    if not rows:
        print("There are no rows to save to CSV (did the experiments fail?).")
        return

    fieldnames = list(rows[0].keys())
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results in CSV format saved to {path}")



def plot_results(all_histories, output_dir="plots_imdb"):
    """
     Generates loss/accuracy plots per experiment.
    If matplotlib is not installed, it prints a message and continues.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # files only; the Tk backend crashes when several runs share a process
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")

        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for hist in all_histories:
        name = hist["experiment"]
        epochs = hist["epoch"]

        # Loss
        plt.figure()
        plt.plot(epochs, hist["train_loss"], label="Train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss - {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"loss_{name}.png")
        plt.close()

        # Accuracy
        plt.figure()
        plt.plot(epochs, hist["train_acc"], label="Train acc")
        plt.plot(epochs, hist["val_acc"], label="Val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy - {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"acc_{name}.png")
        plt.close()

    print(f"Plots saved in folder {out_dir}")


# ---------------------------------------------------------------------------
# Main: runs the experiments with different configurations
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "small",
        "seq_len": 64,
        "batch_size": 32,
        "d_model": 64,
        "num_layers": 1,
        "num_heads": 2,
        "d_ff": 256,
        "dropout": 0.1,
        "num_epochs": 2,
        "lr": 1e-4,
    },
    {
        "name": "base",
        "seq_len": 128,
        "batch_size": 32,
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "num_epochs": 2,
        "lr": 1e-4,
    },
    {
        "name": "wider",
        "seq_len": 128,
        "batch_size": 32,
        "d_model": 256,
        "num_layers": 2,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "num_epochs": 2,
        "lr": 1e-4,
    },
    {
        "name": "deep",
        "seq_len": 128,
        "batch_size": 32,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "num_epochs": 2,
        "lr": 1e-4,
    },
    {
        "name": "longer_seq",
        "seq_len": 256,
        "batch_size": 16,
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "num_epochs": 2,
        "lr": 1e-4,
    },
]


def run_all_experiments(
    names=None,
    init: str = "default",
    seed=None,
    epochs=None,
    out_dir: Path = Path("runs"),
    tokenizer_file: str = "imdb_tokenizer.json",
):
    known = [e["name"] for e in EXPERIMENTS]
    names = names or known
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise SystemExit(f"Unknown experiments: {unknown}. Choose from {known}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) We load IMDb from HuggingFace only once
    raw_ds = load_dataset("imdb")
    train_raw = raw_ds["train"]

    # 2) Tokenizer (only once, reused in all experiments)
    tokenizer = build_tokenizer_imdb(tokenizer_file, imdb_text_iterator(train_raw))

    print("Raw train size:", len(raw_ds["train"]))
    print("Raw test size:", len(raw_ds["test"]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_histories = []

    # 3) Train every requested configuration
    for base_cfg in EXPERIMENTS:
        if base_cfg["name"] not in names:
            continue
        cfg = dict(base_cfg, init=init, seed=seed)
        if epochs is not None:
            cfg["num_epochs"] = epochs
        if seed is not None:
            torch.manual_seed(seed)
        hist = train_one_experiment(
            name=cfg["name"],
            device=device,
            train_raw=train_raw,
            tokenizer=tokenizer,
            config=cfg,
            init=init,
            checkpoint_dir=out_dir / "checkpoints",
        )
        all_histories.append(hist)
        # saved after every experiment, so an interrupted run keeps its results
        save_results_json(all_histories, out_dir / "results_imdb_transformer.json")
        save_results_csv(all_histories, out_dir / "results_imdb_transformer.csv")

    plot_results(all_histories, out_dir / "plots")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", help="subset of configurations (default: all five)")
    p.add_argument("--init", choices=list(INITS), default="default")
    p.add_argument("--seed", type=int, help="seed for weights and shuffling (the original runs used none)")
    p.add_argument("--epochs", type=int, help="override the number of epochs")
    p.add_argument("--out-dir", default="runs")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_all_experiments(args.experiments, args.init, args.seed, args.epochs, args.out_dir)
