"""
Cross-Dataset Generalization Chart (Week 19)
===========================================

Plots a simple bar chart for cross-dataset evaluation results.

This script is designed for *real-only* datasets (e.g. Survey369),
therefore the key metric is the False Positive Rate (FPR):
how often real user videos are incorrectly predicted as fake.

Input:
- experiments/cross_dataset/metrics_summary.csv

Output:
- experiments/cross_dataset/cross_test_fpr.png

Author: Giulio Dajani
Project: VerifAI – Multimodal Deepfake Detection Framework
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="experiments/cross_dataset/metrics_summary.csv",
        help="Path to metrics_summary.csv"
    )
    parser.add_argument(
        "--out",
        default="experiments/cross_dataset/cross_test_fpr.png",
        help="Output chart path"
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    df = pd.read_csv(args.metrics)
    values = {row["metric"]: float(row["value"]) for _, row in df.iterrows()}

    fpr = values.get("false_positive_rate", None)
    avg_prob = values.get("average_prob_fake", None)

    if fpr is None:
        raise ValueError("false_positive_rate not found in metrics_summary.csv")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Plot
    plt.figure()
    plt.bar(["Survey369 (real-only)"], [fpr])
    plt.ylabel("False Positive Rate (FPR)")
    plt.title("Cross-Dataset Generalization: Real-Only Evaluation")
    plt.ylim(0, 1)

    # Annotate
    label = f"FPR={fpr:.3f}"
    if avg_prob is not None:
        label += f"\nAvg prob_fake={avg_prob:.3f}"
    plt.text(0, fpr + 0.02, label, ha="center")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Wrote chart: {args.out}")


if __name__ == "__main__":
    main()
