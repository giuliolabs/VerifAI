"""
Cross-Dataset Generalization Chart (Week 19)
===========================================

This script generates a simple bar chart visualizing the cross-dataset
evaluation results for the VerifAI deepfake detection framework.

In this experiment, the model is evaluated on a *real-only* dataset
(Survey369). Since the dataset contains only genuine videos, the key
evaluation metric is the False Positive Rate (FPR):

    FPR = proportion of real videos incorrectly classified as fake.

A low FPR indicates good generalization to unseen real-world videos.

------------------------------------------------
INPUT
------------------------------------------------
CSV file containing evaluation metrics:

    experiments/cross_dataset/metrics_summary.csv

Expected format:

    metric,value
    false_positive_rate,0.08
    average_prob_fake,0.12

------------------------------------------------
OUTPUT
------------------------------------------------
A bar chart is saved to:

    experiments/cross_dataset/cross_test_fpr.png

The chart shows:
- False Positive Rate
- Optional average fake probability annotation

------------------------------------------------
DEPENDENCIES
------------------------------------------------
Required packages:

    pip install pandas matplotlib

------------------------------------------------
RUN
------------------------------------------------
Example usage:

python scripts/plot_cross_dataset.py \
    --metrics experiments/cross_dataset/metrics_summary.csv \
    --out experiments/cross_dataset/cross_test_fpr.png

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
This visualization supports Week 19 cross-dataset generalization
analysis by demonstrating how the trained model behaves on an unseen
real-world dataset (Survey369). Since the dataset contains only real
videos, the FPR provides the most meaningful indicator of model
robustness.

Author: Giulio Dajani 001343717
Project: VerifAI – Multimodal Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# -------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------

# os is used for file existence checks and directory creation
import os

# argparse allows command-line arguments
import argparse

# -------------------------------------------------------
# Third-Party Imports
# -------------------------------------------------------

# pandas reads the metrics CSV file
import pandas as pd

# matplotlib generates the visualization
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Main function
# -------------------------------------------------------
def main() -> None:
    """
    Generate a cross-dataset evaluation chart.

    The script reads metrics from a CSV file and produces
    a bar chart showing the false positive rate (FPR)
    on a real-only dataset.
    """

    # ---------------------------------------------
    # Parse command-line arguments
    # ---------------------------------------------
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

    # ---------------------------------------------
    # Validate input file
    # ---------------------------------------------
    if not os.path.exists(args.metrics):
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    # ---------------------------------------------
    # Load metrics CSV
    # ---------------------------------------------
    df = pd.read_csv(args.metrics)

    # Convert CSV rows into a dictionary for easier lookup
    values = {row["metric"]: float(row["value"]) for _, row in df.iterrows()}

    # Extract required metrics
    fpr = values.get("false_positive_rate", None)
    avg_prob = values.get("average_prob_fake", None)

    if fpr is None:
        raise ValueError("false_positive_rate not found in metrics_summary.csv")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ---------------------------------------------
    # Generate bar chart
    # ---------------------------------------------
    plt.figure()

    # Plot single-bar chart (Survey369 real-only dataset)
    plt.bar(["Survey369 (real-only)"], [fpr])

    plt.ylabel("False Positive Rate (FPR)")
    plt.title("Cross-Dataset Generalization: Real-Only Evaluation")

    # FPR is a probability so range is [0,1]
    plt.ylim(0, 1)

    # ---------------------------------------------
    # Annotate chart with metric values
    # ---------------------------------------------
    label = f"FPR={fpr:.3f}"

    if avg_prob is not None:
        label += f"\nAvg prob_fake={avg_prob:.3f}"

    plt.text(0, fpr + 0.02, label, ha="center")

    plt.tight_layout()

    # Save chart
    plt.savefig(args.out, dpi=200)

    plt.close()

    print(f"Wrote chart: {args.out}")


# -------------------------------------------------------
# Script entry point
# -------------------------------------------------------
if __name__ == "__main__":
    main()