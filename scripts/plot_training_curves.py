"""
Training Curve Plot Generator – VerifAI
=======================================

This script reads a CSV file containing training metrics (e.g., loss,
accuracy, AUC) and generates plots showing how these metrics change
across training epochs.

It is used for visualizing model training progress and saving plots that
can be included in experiment reports, ablation studies, or thesis
figures.

Typical CSV inputs come from training scripts that log metrics such as:

    epoch,train_loss,val_loss,val_auc,val_f1,train_acc,val_acc

The script automatically detects which metrics exist in the CSV and
generates the corresponding plots.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
Required packages:

    pip install matplotlib

------------------------------------------------
RUN
------------------------------------------------
Example usage:

python scripts/plot_training_curves.py \
    --csv experiments/logs/training_curves/xception_run1.csv

Optional output directory:

python scripts/plot_training_curves.py \
    --csv experiments/logs/training_curves/xception_run1.csv \
    --outdir experiments/logs/training_curves/plots

------------------------------------------------
OUTPUT
------------------------------------------------
Generated plots are saved as PNG files:

    <csv_name>_loss.png
    <csv_name>_val_auc.png
    <csv_name>_val_f1.png
    <csv_name>_accuracy.png

These plots help visualize model convergence and validation performance.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# -------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------

# argparse allows command-line arguments
import argparse

# pathlib handles file paths in a platform-independent way
from pathlib import Path

# csv reads training logs stored as CSV files
import csv

# math is used for NaN checking
import math

# -------------------------------------------------------
# Third-Party Imports
# -------------------------------------------------------

# matplotlib is used to generate training curves
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Helper function: read CSV file
# -------------------------------------------------------
def read_csv(path: Path) -> list[dict]:
    """
    Read a CSV file and return rows as dictionaries.

    Each row corresponds to one training epoch and contains
    metric values such as loss, accuracy, or AUC.
    """
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# -------------------------------------------------------
# Helper function: safe float conversion
# -------------------------------------------------------
def to_float(value: str) -> float:
    """
    Convert a string value to float.

    If conversion fails, return NaN so the plotting logic
    can safely skip invalid values.
    """
    try:
        return float(value)
    except Exception:
        return float("nan")


# -------------------------------------------------------
# Main plotting routine
# -------------------------------------------------------
def main() -> None:

    parser = argparse.ArgumentParser(
        description="Generate training curve plots from a CSV log."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to training curve CSV file"
    )

    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for plots"
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------
    csv_path = Path(args.csv)

    # If no output directory is provided, use the CSV directory
    outdir = Path(args.outdir) if args.outdir else csv_path.parent

    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Load CSV data
    # -------------------------------------------------------
    rows = read_csv(csv_path)

    if not rows:
        raise SystemExit("No data found in CSV.")

    # -------------------------------------------------------
    # Parse epoch values
    # -------------------------------------------------------
    # Some logs may not contain an explicit epoch column,
    # so we fall back to row index if needed.
    epochs = []

    for i, r in enumerate(rows):

        e = r.get("epoch", "")

        if isinstance(e, str) and e.strip().isdigit():
            epochs.append(int(e))
        else:
            epochs.append(i + 1)

    # -------------------------------------------------------
    # Plot helper
    # -------------------------------------------------------
    def plot_if_present(keys: list[str], ylabel: str, title_suffix: str, out_suffix: str) -> None:
        """
        Plot one or more metrics if they exist in the CSV.

        Args:
            keys:
                column names to plot

            ylabel:
                label for the y-axis

            title_suffix:
                title prefix

            out_suffix:
                suffix used for output file name
        """

        # Check if any requested key exists in CSV columns
        present = any(k in rows[0] for k in keys)

        if not present:
            return

        plt.figure()

        for k in keys:

            if k in rows[0]:

                ys = [to_float(r.get(k, "")) for r in rows]

                # Skip metrics that contain only NaN values
                if all(math.isnan(v) for v in ys):
                    continue

                plt.plot(epochs, ys, label=k)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)

        plt.title(f"{title_suffix}: {csv_path.stem}")

        plt.legend()

        out = outdir / f"{csv_path.stem}_{out_suffix}.png"

        plt.savefig(out, dpi=200, bbox_inches="tight")

        plt.close()

        print("[OK] Saved:", out)

    # -------------------------------------------------------
    # Generate plots
    # -------------------------------------------------------

    # Loss curves
    plot_if_present(
        ["train_loss", "val_loss"],
        "Loss",
        "Loss Curves",
        "loss",
    )

    # Validation AUC
    plot_if_present(
        ["val_auc"],
        "AUC",
        "Validation AUC",
        "val_auc",
    )

    # Validation F1 score
    plot_if_present(
        ["val_f1"],
        "F1",
        "Validation F1",
        "val_f1",
    )

    # Accuracy curves
    plot_if_present(
        ["acc", "train_acc", "val_acc"],
        "Accuracy",
        "Accuracy",
        "accuracy",
    )


# -------------------------------------------------------
# Script entry point
# -------------------------------------------------------
if __name__ == "__main__":
    main()