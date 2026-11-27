"""How much does weight initialisation matter for the `base` configuration?

Trains every (initialisation, seed) pair that has not been trained yet, scores each
checkpoint on the validation split and on the official test split, and writes
results/init_ablation.csv. Finished runs are reused, so it can be interrupted.

    python ablation.py
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from evaluate import load_checkpoint, predict
from train_imdb import (
    EXPERIMENTS,
    INITS,
    ImdbDataset,
    build_tokenizer_imdb,
    imdb_text_iterator,
    run_all_experiments,
)

CONFIG = "base"
SEEDS = (0, 1, 2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="results/init_ablation.csv")
    p.add_argument("--tokenizer", default="imdb_tokenizer.json")
    args = p.parse_args()

    cfg = next(e for e in EXPERIMENTS if e["name"] == CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_dataset("imdb")
    tokenizer = build_tokenizer_imdb(args.tokenizer, imdb_text_iterator(raw["train"]))
    test_dl = DataLoader(ImdbDataset(raw["test"], tokenizer, cfg["seq_len"]), batch_size=64)

    rows = []
    for init in INITS:
        for seed in SEEDS:
            run = Path(args.runs_dir) / f"init_{init}_s{seed}"
            history = run / "results_imdb_transformer.csv"
            checkpoint = run / "checkpoints" / f"encoder_imdb_{CONFIG}.pt"
            if not (history.exists() and checkpoint.exists()):
                run_all_experiments([CONFIG], init, seed, None, run, args.tokenizer)

            last = pd.read_csv(history).sort_values("epoch").iloc[-1]
            model = load_checkpoint(checkpoint, cfg, tokenizer.get_vocab_size()).to(device)
            pred, y = predict(model, test_dl, device)
            rows.append(
                {
                    "init": init,
                    "seed": seed,
                    "epochs": int(last["epoch"]),
                    "train_loss": last["train_loss"],
                    "train_acc": last["train_acc"],
                    "val_acc": last["val_acc"],
                    "test_acc": (pred == y).float().mean().item(),
                }
            )
            print(rows[-1], flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
