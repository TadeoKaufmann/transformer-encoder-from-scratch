"""Bag-of-words reference point: TF-IDF (unigrams and bigrams) + logistic regression.

It trains on exactly the same 22,500 reviews as the Transformer and is scored on the
same validation split and on the official test split.

    python baseline.py
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from torch.utils.data import random_split


def split_indices(n: int):
    """The 90/10 split used by train_imdb.create_dataloaders (seed 42)."""
    train_size = int(0.9 * n)
    train, val = random_split(
        range(n), [train_size, n - train_size], generator=torch.Generator().manual_seed(42)
    )
    return list(train.indices), list(val.indices)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    raw = load_dataset("imdb")
    texts, labels = raw["train"]["text"], raw["train"]["label"]
    train_idx, val_idx = split_indices(len(texts))

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=10.0, max_iter=1000),
    )
    start = time.time()
    model.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
    fit_time = time.time() - start

    row = {
        "model": "tfidf_logreg",
        "val_acc": model.score([texts[i] for i in val_idx], [labels[i] for i in val_idx]),
        "test_acc": model.score(raw["test"]["text"], raw["test"]["label"]),
        "fit_time_s": fit_time,
    }
    print(row)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out / "baseline.csv", index=False)


if __name__ == "__main__":
    main()
