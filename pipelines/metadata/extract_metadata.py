"""
Generic Metadata Feature Extraction
===================================

This script builds a simple numeric feature vector for each row in a
dataset split CSV. It is designed to be dataset-agnostic, meaning it
works with any split file as long as some common columns are present.

The idea is to convert lightweight metadata into fixed numeric vectors
that can later be used for:
- baseline models
- metadata analysis
- ablation experiments
- reproducibility checks

INPUT
-----
Split CSV:
    data/splits/<dataset>_<split>.csv

Typical useful columns:
    video_id
    video_path
    label
    method
    compression

OUTPUT
------
Feature files:
    data/processed/metadata_features/<Dataset>/<split>/<safe_id>.npy

Also writes:
    data/processed/metadata_features/<Dataset>/<split>/_columns.txt

The `_columns.txt` file records the exact feature order so the saved
vectors remain perfectly reproducible and interpretable.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

import argparse
from pathlib import Path

# NumPy is used to store the final feature vectors as .npy files
import numpy as np

# pandas is used to read the split CSV file
import pandas as pd

# tqdm adds a progress bar during extraction
from tqdm import tqdm

# hashlib is used to create short, reproducible safe IDs from video IDs
import hashlib


def make_safe_id(video_id: str) -> str:
    """
    Convert a video_id into a short deterministic identifier.

    This is useful when saving features on Windows systems because it
    keeps filenames short and reproducible.
    """
    return hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:16]


def normalize_label(value) -> float:
    """
    Convert common label formats into numeric form.

    Supported examples:
      - real / fake
      - 0 / 1
      - true / false
      - yes / no
      - authentic / deepfake

    Returns:
        0.0 for real-like labels
        1.0 for fake-like labels

    If the label is unknown, the function returns 0.0 conservatively.
    """
    if value is None:
        return 0.0

    # If the value is already numeric, keep it
    try:
        return float(value)
    except Exception:
        pass

    s = str(value).strip().lower()

    if s in {"real", "bonafide", "genuine", "negative", "authentic"}:
        return 0.0

    if s in {"fake", "spoof", "forged", "positive", "deepfake"}:
        return 1.0

    if s in {"true", "yes"}:
        return 1.0

    if s in {"false", "no"}:
        return 0.0

    # Default fallback
    return 0.0


def main():
    """
    Main execution logic.

    The script:
    1. Loads the split CSV
    2. Builds stable vocabularies for categorical columns
    3. Saves the feature order to _columns.txt
    4. Creates one numeric feature vector per sample
    5. Saves each feature vector as <safe_id>.npy
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build categorical vocabularies using sorted unique values
    # Sorting keeps the one-hot encoding order stable and reproducible.
    method_vocab = (
        sorted(df["method"].dropna().astype(str).unique().tolist())
        if "method" in df.columns else []
    )

    comp_vocab = (
        sorted(df["compression"].dropna().astype(str).unique().tolist())
        if "compression" in df.columns else []
    )

    # Record the feature order exactly as used in the vectors
    feature_names = ["label"]

    for m in method_vocab:
        feature_names.append(f"method__{m}")

    for c in comp_vocab:
        feature_names.append(f"compression__{c}")

    # Add simple numeric metadata features
    feature_names += ["path_len", "filename_len"]

    # Save feature order for perfect reproducibility
    (out_dir / "_columns.txt").write_text("\n".join(feature_names), encoding="utf-8")

    saved = 0
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Metadata feats -> {out_dir}"):
        video_id = str(row["video_id"])
        safe_id = make_safe_id(video_id)
        out_path = out_dir / f"{safe_id}.npy"

        # Resume-safe: skip if this feature file already exists
        if out_path.exists():
            skipped += 1
            continue

        # Read available fields, using safe defaults if missing
        label = normalize_label(row["label"]) if "label" in df.columns else 0.0
        method = str(row["method"]) if "method" in df.columns else ""
        comp = str(row["compression"]) if "compression" in df.columns else ""
        vpath = str(row["video_path"]) if "video_path" in df.columns else ""

        feat = [label]

        # One-hot encode method column
        for m in method_vocab:
            feat.append(1.0 if method == m else 0.0)

        # One-hot encode compression column
        for c in comp_vocab:
            feat.append(1.0 if comp == c else 0.0)

        # Add very simple deterministic numeric metadata
        feat.append(float(len(vpath)))               # full path length
        feat.append(float(len(Path(vpath).name)))    # filename length only

        np.save(out_path, np.array(feat, dtype=np.float32))
        saved += 1

    print("\nDone.")
    print("Saved:", saved)
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()