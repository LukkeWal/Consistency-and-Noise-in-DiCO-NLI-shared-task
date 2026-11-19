"""Create plots for a training run stored in results.csv."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DataHandler import LABELS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results.csv"
OUTPUT_DIR = ROOT / "visualizations"


def read_result(result_id):
    with RESULTS.open(encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row.get("id") == str(result_id)]
    if len(rows) != 1:
        raise ValueError(f"Expected one result with ID {result_id}, found {len(rows)}")
    return rows[0]


def create_plot(result_id, plot):
    row = read_result(result_id)
    field = "loss_history" if plot == "loss" else "confusion_matrix"
    if not row.get(field):
        raise ValueError(f"Result {result_id} has no {field} data")
    data = json.loads(row[field])
    if not data:
        raise ValueError(f"Result {result_id} has no {field} data")

    if plot == "loss":
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = [entry["epoch"] for entry in data]
        ax.plot(epochs, [entry["train_loss"] for entry in data], marker="o", label="Training")
        ax.plot(epochs, [entry["val_loss"] for entry in data], marker="o", label="Validation")
        ax.set(xlabel="Epoch", ylabel="Loss", title=f"Training and validation loss — result {result_id}")
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        matrix = [[data[actual][predicted] for predicted in LABELS] for actual in LABELS]
        fig, ax = plt.subplots(figsize=(9, 7))
        heatmap = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(heatmap, ax=ax, label="Count")
        ax.set_xticks(range(len(LABELS)), labels=LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(LABELS)), labels=LABELS)
        ax.set(xlabel="Predicted label", ylabel="Actual label", title=f"Confusion matrix — result {result_id}")
        threshold = max(max(row) for row in matrix) / 2
        for i, values in enumerate(matrix):
            for j, value in enumerate(values):
                ax.text(j, i, str(value), ha="center", va="center", color="white" if value > threshold else "black")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / f"result_{result_id}_{plot}.png"
    try:
        fig.savefig(target, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser(description="Plot a training result from results.csv")
    parser.add_argument("id", type=int, help="Incremental result ID")
    parser.add_argument("plot", choices=("loss", "confusion-matrix"), help="Plot to generate")
    args = parser.parse_args()
    if args.id < 1:
        parser.error("Result ID must be positive")
    try:
        path = create_plot(args.id, args.plot)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
