"""Draw the README figures from the CSV files in results/.

    python make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
FIGURES = Path("figures")

BLUE, ORANGE, AQUA, GREY = "#257de0", "#f47230", "#00b894", "#9a9a9a"
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"

INIT_LABELS = {
    "default": "PyTorch defaults\n(as submitted)",
    "xavier": "Xavier on\nevery matrix",
    "xavier_embedding": "Xavier on the\nembedding only",
    "xavier_layers": "Xavier on everything\nbut the embedding",
}

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
    }
)


def test_accuracy():
    models = pd.read_csv(RESULTS / "test_evaluation.csv")
    baseline = pd.read_csv(RESULTS / "baseline.csv").iloc[0]
    ablation = pd.read_csv(RESULTS / "init_ablation.csv")
    xavier = ablation[ablation["init"] == "xavier"]["test_acc"].mean()

    bars = [(f"{r.experiment}", r.test_acc, BLUE) for r in models.itertuples()]
    bars.append(("base, Xavier init\n(mean of 3 seeds)", xavier, ORANGE))
    bars.append(("TF-IDF +\nlogistic regression", baseline["test_acc"], GREY))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for y, (_, value, colour) in enumerate(bars):
        ax.barh(y, value, color=colour, height=0.62)
        ax.text(value + 0.008, y, f"{value:.3f}", va="center", color=INK)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(0.5, -0.75, "chance", color=MUTED, ha="center", va="center")
    ax.set_xlabel("Accuracy on the 25,000 test reviews (2 epochs for the Transformers)")
    ax.set_title("Test accuracy", pad=14)
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIGURES / "test_accuracy.png")
    plt.close(fig)


def accuracy_by_length():
    table = pd.read_csv(RESULTS / "test_accuracy_by_length.csv")
    series = [("base", "base (128 tokens)", BLUE), ("wider", "wider (128 tokens)", ORANGE),
              ("longer_seq", "longer_seq (256 tokens)", AQUA)]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, label, colour in series:
        part = table[table["experiment"] == name]
        ax.plot(range(len(part)), part["accuracy"], color=colour, linewidth=2, marker="o", markersize=6,
                label=label)
    first = table[table["experiment"] == "base"]
    ax.set_xticks(range(len(first)), [f"{r.bucket}\nn={r.n:,}" for r in first.itertuples()])
    ax.set_xlabel("Review length in word-level tokens")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.6, 0.85)
    ax.grid(axis="y", color=GRID)
    ax.set_title("Longer reviews are harder")
    ax.legend()
    fig.savefig(FIGURES / "accuracy_by_length.png")
    plt.close(fig)


def init_ablation():
    table = pd.read_csv(RESULTS / "init_ablation.csv")
    order = [k for k in INIT_LABELS if k in set(table["init"])]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    for x, init in enumerate(order):
        values = table[table["init"] == init]["test_acc"]
        offsets = [x + 0.07 * (i - (len(values) - 1) / 2) for i in range(len(values))]
        ax.scatter(offsets, values, s=42, color=BLUE, edgecolor="white", linewidth=1, zorder=3)
        ax.text(x + 0.2, values.mean(), f"{values.mean():.3f}", va="center", color=INK)
    ax.set_xticks(range(len(order)), [INIT_LABELS[k] for k in order])
    ax.set_xlim(-0.5, len(order) - 0.4)
    ax.set_ylim(0.65, 0.9)
    ax.set_ylabel("Test accuracy")
    ax.grid(axis="y", color=GRID)
    ax.set_title("base configuration: one dot per seed, number = mean")
    fig.savefig(FIGURES / "init_ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    test_accuracy()
    accuracy_by_length()
    init_ablation()
    print(f"figures written to {FIGURES}")
