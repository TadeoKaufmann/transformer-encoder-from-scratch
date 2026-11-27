# make_better_plots.py
# Usage:
#   python make_better_plots.py --csv results_imdb_transformer.csv --out better_plots
# or
#   python make_better_plots.py --json results_imdb_transformer.json --out better_plots

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_results(csv_path: str | None, json_path: str | None) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
        return df
    if json_path:
        # JSON is a list of histories; convert to long DF compatible with your CSV format
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            histories = json.load(f)

        rows = []
        for h in histories:
            name = h["experiment"]
            cfg = h["config"]
            total_time = h.get("total_time", None)

            for i, epoch in enumerate(h["epoch"]):
                rows.append({
                    "experiment": name,
                    "epoch": epoch,
                    "train_loss": h["train_loss"][i],
                    "train_acc": h["train_acc"][i],
                    "val_acc": h["val_acc"][i],
                    "epoch_time": h["epoch_time"][i],
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

        return pd.DataFrame(rows)

    raise ValueError("Provide --csv or --json")


def final_table(df: pd.DataFrame) -> pd.DataFrame:
    # One row per experiment: last epoch
    df = df.copy()
    df["epoch"] = df["epoch"].astype(int)
    df_last = df.sort_values(["experiment", "epoch"]).groupby("experiment", as_index=False).tail(1)

    # Keep columns useful for report table + plots
    cols = [
        "experiment", "seq_len", "batch_size", "d_model", "num_layers", "num_heads", "d_ff",
        "train_loss", "train_acc", "val_acc", "total_time"
    ]
    df_last = df_last[cols].sort_values("val_acc", ascending=False).reset_index(drop=True)
    return df_last


def save_table_as_csv(df_last: pd.DataFrame, out_dir: Path):
    out_path = out_dir / "final_summary_table.csv"
    df_last.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def plot_curves_together(df: pd.DataFrame, out_dir: Path):
    # Same axes, all experiments in one plot (comparison plot)
    for metric, ylabel, title, fname in [
        ("train_loss", "Loss", "Train Loss (all experiments)", "curves_train_loss_all.png"),
        ("train_acc", "Accuracy", "Train Accuracy (all experiments)", "curves_train_acc_all.png"),
        ("val_acc", "Accuracy", "Validation Accuracy (all experiments)", "curves_val_acc_all.png"),
    ]:
        plt.figure()
        for exp, g in df.groupby("experiment"):
            g = g.sort_values("epoch")
            plt.plot(g["epoch"], g[metric], marker="o", label=exp)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()
        print(f"Saved: {out_dir / fname}")


def plot_fixed_axes_per_experiment(df: pd.DataFrame, out_dir: Path):
    # Fix y-limits globally so individual plots are visually comparable.
    # (This solves your "all look the same" issue.)
    loss_min, loss_max = df["train_loss"].min(), df["train_loss"].max()
    acc_min = min(df["train_acc"].min(), df["val_acc"].min())
    acc_max = max(df["train_acc"].max(), df["val_acc"].max())

    # Add small padding so lines don't touch borders
    loss_pad = 0.05 * (loss_max - loss_min + 1e-9)
    acc_pad = 0.05 * (acc_max - acc_min + 1e-9)

    y_loss = (loss_min - loss_pad, loss_max + loss_pad)
    y_acc = (acc_min - acc_pad, acc_max + acc_pad)

    for exp, g in df.groupby("experiment"):
        g = g.sort_values("epoch")

        # Loss
        plt.figure()
        plt.plot(g["epoch"], g["train_loss"], marker="o", label="Train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(*y_loss)
        plt.title(f"Loss (fixed y) - {exp}")
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"loss_fixed_{exp}.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")

        # Accuracy
        plt.figure()
        plt.plot(g["epoch"], g["train_acc"], marker="o", label="Train acc")
        plt.plot(g["epoch"], g["val_acc"], marker="o", label="Val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(*y_acc)
        plt.title(f"Accuracy (fixed y) - {exp}")
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"acc_fixed_{exp}.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")


def plot_valacc_vs_time(df_last: pd.DataFrame, out_dir: Path):
    # Scatter plot: summarizes the trade-off perfectly for the report
    plt.figure()
    plt.scatter(df_last["total_time"], df_last["val_acc"])

    for _, row in df_last.iterrows():
        plt.annotate(
            row["experiment"],
            (row["total_time"], row["val_acc"]),
            textcoords="offset points",
            xytext=(6, 4),
        )

    plt.xlabel("Total training time (s)")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy vs training time")
    plt.tight_layout()
    path = out_dir / "scatter_valacc_vs_time.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_bars(df_last: pd.DataFrame, out_dir: Path):
    # Bar charts: easy to read in a report
    order = df_last.sort_values("val_acc", ascending=False)["experiment"].tolist()

    # Val acc bar
    plt.figure()
    plt.bar(df_last.set_index("experiment").loc[order].index,
            df_last.set_index("experiment").loc[order]["val_acc"])
    plt.xlabel("Experiment")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy by experiment")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = out_dir / "bar_val_acc.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

    # Time bar
    plt.figure()
    plt.bar(df_last.set_index("experiment").loc[order].index,
            df_last.set_index("experiment").loc[order]["total_time"])
    plt.xlabel("Experiment")
    plt.ylabel("Total training time (s)")
    plt.title("Training time by experiment")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = out_dir / "bar_total_time.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to results_imdb_transformer.csv")
    parser.add_argument("--json", type=str, default=None, help="Path to results_imdb_transformer.json")
    parser.add_argument("--out", type=str, default="better_plots", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.csv, args.json)
    df_last = final_table(df)

    save_table_as_csv(df_last, out_dir)

    # Plots
    plot_curves_together(df, out_dir)
    plot_fixed_axes_per_experiment(df, out_dir)
    plot_valacc_vs_time(df_last, out_dir)
    plot_bars(df_last, out_dir)

    print("\nDone. Recommended for report:")
    print("- scatter_valacc_vs_time.png")
    print("- curves_val_acc_all.png")
    print("- (optional) bar_val_acc.png and bar_total_time.png")
    print("- per-experiment fixed-y plots if you still want one figure per model.")


if __name__ == "__main__":
    main()
