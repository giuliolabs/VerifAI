"""
Main User Study Analysis – SUS + Trust Metrics (Week 20)
======================================================

Computes usability and trust metrics for the VerifAI main user study (~30 participants).

This script:
1) Loads anonymized survey responses
2) Computes SUS (System Usability Scale) score per participant
3) Computes a simple Trust Index (mean of trust items)
4) Outputs:
   - experiments/user_study/analysis/sus_trust_summary.csv
   - experiments/user_study/analysis/sus_trust_plots.png

SUS Scoring (standard):
- Odd items (1,3,5,7,9):   contribution = response - 1
- Even items (2,4,6,8,10): contribution = 5 - response
- SUS total = sum(contributions) * 2.5  -> range 0..100

Trust Index (project-defined):
- Average of: trust_understand, trust_confidence, trust_reliance,
              trust_transparency, trust_overall  -> range 1..5

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = os.path.join("experiments", "user_study", "surveys", "responses_anonymized.csv")
OUT_SUMMARY = os.path.join("experiments", "user_study", "analysis", "sus_trust_summary.csv")
OUT_PLOT = os.path.join("experiments", "user_study", "analysis", "sus_trust_plots.png")


SUS_ITEMS = ["sus_q1","sus_q2","sus_q3","sus_q4","sus_q5","sus_q6","sus_q7","sus_q8","sus_q9","sus_q10"]
TRUST_ITEMS = ["trust_understand","trust_confidence","trust_reliance","trust_transparency","trust_overall"]


def compute_sus(row: pd.Series) -> float:
    # SUS expects integer 1..5
    responses = [int(row[item]) for item in SUS_ITEMS]

    total = 0
    for idx, value in enumerate(responses, start=1):
        if idx % 2 == 1:
            total += (value - 1)
        else:
            total += (5 - value)

    return float(total) * 2.5


def main() -> None:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing input: {INPUT_CSV}")

    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # Basic validation
    for col in SUS_ITEMS + TRUST_ITEMS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute scores
    df["sus_score"] = df.apply(compute_sus, axis=1)
    df["trust_index"] = df[TRUST_ITEMS].mean(axis=1)

    # Summary table (keep it tidy for examiners)
    summary_cols = [
        "participant_id", "device", "task_time_sec", "task_success", "errors_count",
        "sus_score", "trust_index",
        "aware_heard_deepfake", "aware_encountered", "aware_concern", "aware_trust_ai_tool",
        "comments"
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    out_df = df[summary_cols].copy()
    out_df.to_csv(OUT_SUMMARY, index=False)

    # Simple plots
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
