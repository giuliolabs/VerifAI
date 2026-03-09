"""
Main User Study Analysis – SUS + Trust Metrics (Week 20)
========================================================

This script performs statistical analysis on the VerifAI main user study
dataset (~30 participants). It computes usability and trust metrics
based on standard SUS scoring and a project-defined Trust Index.

The output includes:
- A cleaned summary CSV for reporting
- A distribution plot of SUS scores

SUS Scoring (standard method):
- Odd items (1,3,5,7,9):   contribution = response - 1
- Even items (2,4,6,8,10): contribution = 5 - response
- Final SUS score = sum(contributions) * 2.5  (range 0..100)

Trust Index (project-defined):
- Mean of trust-related Likert items (range 1..5)

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Used for file path handling and directory checks
import os

# Pandas for structured survey data processing
import pandas as pd

# Matplotlib for simple statistical visualization
import matplotlib.pyplot as plt


# ------------------------------------------------
# File Paths
# ------------------------------------------------

# Input: anonymized survey dataset
INPUT_CSV = os.path.join("experiments", "user_study", "surveys", "responses_anonymized.csv")

# Output: cleaned summary table
OUT_SUMMARY = os.path.join("experiments", "user_study", "analysis", "sus_trust_summary.csv")

# Output: SUS distribution plot
OUT_PLOT = os.path.join("experiments", "user_study", "analysis", "sus_trust_plots.png")


# ------------------------------------------------
# Column Definitions
# ------------------------------------------------

# Standard 10 SUS questionnaire items
SUS_ITEMS = [
    "sus_q1","sus_q2","sus_q3","sus_q4","sus_q5",
    "sus_q6","sus_q7","sus_q8","sus_q9","sus_q10"
]

# Project-defined trust dimensions
TRUST_ITEMS = [
    "trust_understand",
    "trust_confidence",
    "trust_reliance",
    "trust_transparency",
    "trust_overall"
]

# ------------------------------------------------
# SUS Computation
# ------------------------------------------------

def compute_sus(row: pd.Series) -> float:
    """
    Compute SUS score for a single participant.

    The SUS formula alternates scoring:
    - Odd items are positively worded
    - Even items are negatively worded

    The final score is scaled to 0–100.
    """

    # Convert responses to integers (expected range 1..5)
    responses = [int(row[item]) for item in SUS_ITEMS]

    total = 0

    # Apply SUS scoring formula
    for idx, value in enumerate(responses, start=1):
        if idx % 2 == 1:
            # Positive item contribution
            total += (value - 1)
        else:
            # Negative item contribution
            total += (5 - value)

    # Scale to 0–100 range
    return float(total) * 2.5


# ------------------------------------------------
# Main Analysis Pipeline
# ------------------------------------------------

def main() -> None:

    # Ensure the input dataset exists before processing
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing input: {INPUT_CSV}")

    # Create output directory if it does not already exist
    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)

    # Load survey dataset
    df = pd.read_csv(INPUT_CSV)

    # ------------------------------------------------
    # Basic Validation
    # ------------------------------------------------

    # Ensure all required SUS and Trust columns are present
    for col in SUS_ITEMS + TRUST_ITEMS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ------------------------------------------------
    # Compute Metrics
    # ------------------------------------------------

    # Compute SUS score per participant
    df["sus_score"] = df.apply(compute_sus, axis=1)

    # Compute Trust Index as mean of trust-related Likert items
    df["trust_index"] = df[TRUST_ITEMS].mean(axis=1)

    # ------------------------------------------------
    # Clean Summary Table
    # ------------------------------------------------

    # Select key variables relevant for reporting and examiner review
    summary_cols = [
        "participant_id",
        "device",
        "task_time_sec",
        "task_success",
        "errors_count",
        "sus_score",
        "trust_index",
        "aware_heard_deepfake",
        "aware_encountered",
        "aware_concern",
        "aware_trust_ai_tool",
        "comments"
    ]

    # Keep only columns that exist in dataset
    summary_cols = [c for c in summary_cols if c in df.columns]

    out_df = df[summary_cols].copy()

    # Export cleaned dataset
    out_df.to_csv(OUT_SUMMARY, index=False)

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    # Plot SUS score distribution for quick usability overview
    plt.figure()
    plt.hist(df["sus_score"], bins=10)
    plt.title("SUS Score Distribution (n=30)")
    plt.xlabel("SUS score (0-100)")
    plt.ylabel("Count")
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Wrote: {OUT_SUMMARY}")
    print(f"Wrote: {OUT_PLOT}")


if __name__ == "__main__":
    main()