"""Evaluate saved checkpoints on the validation split and on the official IMDb test split.

Also breaks test accuracy down by review length, to see who benefits from a longer
context window.

    python evaluate.py --checkpoints runs/checkpoints
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from model import TransformerEncoderClassifier
from train_imdb import (
    EXPERIMENTS,
    ImdbDataset,
    build_tokenizer_imdb,
    create_dataloaders,
    imdb_text_iterator,
)

# Review length in word-level tokens. The edges follow the context sizes under study.
LENGTH_BINS = [0, 62, 126, 254, 510, float("inf")]
LENGTH_LABELS = ["1-62", "63-126", "127-254", "255-510", "511+"]


def accuracy_by_length(n_tokens, correct) -> pd.DataFrame:
    """Accuracy and example count for each review-length bucket."""
    df = pd.DataFrame({"n_tokens": n_tokens, "correct": correct})
    df["bucket"] = pd.cut(df["n_tokens"], bins=LENGTH_BINS, labels=LENGTH_LABELS)
    out = df.groupby("bucket", observed=False)["correct"].agg(["mean", "size"]).reset_index()
    return out.rename(columns={"mean": "accuracy", "size": "n"})


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        logits = model(batch["encoder_input"].to(device), batch["encoder_mask"].to(device))
        preds.append(logits.argmax(dim=-1).cpu())
        labels.append(batch["label"])
    return torch.cat(preds), torch.cat(labels)


def load_checkpoint(path, cfg: dict, vocab_size: int) -> TransformerEncoderClassifier:
    """Rebuild the model of one configuration and load its saved weights."""
    model = TransformerEncoderClassifier(
        vocab_size=vocab_size,
        seq_len=cfg["seq_len"],
        num_classes=2,
        d_model=cfg["d_model"],
        N=cfg["num_layers"],
        h=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", default="runs/checkpoints")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--tokenizer", default="imdb_tokenizer.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_dataset("imdb")
    tokenizer = build_tokenizer_imdb(args.tokenizer, imdb_text_iterator(raw["train"]))
    test_lengths = [len(e.ids) for e in tokenizer.encode_batch(raw["test"]["text"])]

    rows, buckets = [], []
    for cfg in EXPERIMENTS:
        name = cfg["name"]
        path = Path(args.checkpoints) / f"encoder_imdb_{name}.pt"
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        model = load_checkpoint(path, cfg, tokenizer.get_vocab_size()).to(device)

        _, val_dl = create_dataloaders(raw["train"], tokenizer, cfg["seq_len"], cfg["batch_size"])
        val_pred, val_y = predict(model, val_dl, device)
        test_dl = DataLoader(ImdbDataset(raw["test"], tokenizer, cfg["seq_len"]), batch_size=64)
        test_pred, test_y = predict(model, test_dl, device)

        n_params = sum(t.numel() for t in model.parameters())
        row = {
            "experiment": name,
            "seq_len": cfg["seq_len"],
            "parameters": n_params,
            "val_acc": (val_pred == val_y).float().mean().item(),
            "test_acc": (test_pred == test_y).float().mean().item(),
            "truncated_test_share": sum(n > cfg["seq_len"] - 2 for n in test_lengths) / len(test_lengths),
        }
        print(json.dumps(row))
        rows.append(row)
        by_len = accuracy_by_length(test_lengths, (test_pred == test_y).numpy())
        by_len.insert(0, "experiment", name)
        buckets.append(by_len)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "test_evaluation.csv", index=False)
    pd.concat(buckets).to_csv(out / "test_accuracy_by_length.csv", index=False)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
