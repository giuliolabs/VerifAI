"""
Generic Metadata Feature Extraction
===================================

Builds a numeric feature vector for each row in a split CSV.
Dataset-agnostic: uses common columns if present.

INPUT
-----
Split CSV:
    data/splits/<dataset>_<split>.csv

OUTPUT
------
Feature files:
    data/processed/metadata_features/<Dataset>/<split>/<safe_id>.npy

Also writes:
    data/processed/metadata_features/<Dataset>/<split>/_columns.txt
so you can see the exact feature order (for perfect reproducibility).

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import hashlib


def make_safe_id(video_id: str) -> str:
    return hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build categorical vocabularies (stable sort for reproducibility)
    method_vocab = sorted(df["method"].dropna().astype(str).unique().tolist()) if "method" in df.columns else []
    comp_vocab = sorted(df["compression"].dropna().astype(str).unique().tolist()) if "compression" in df.columns else []

    feature_names = ["label"]

    for m in method_vocab:
        feature_names.append(f"method__{m}")
    for c in comp_vocab:
        feature_names.append(f"compression__{c}")

    feature_names += ["path_len", "filename_len"]

    (out_dir / "_columns.txt").write_text("\n".join(feature_names), encoding="utf-8")

    saved = skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Metadata feats -> {out_dir}"):
        video_id = str(row["video_id"])
        safe_id = make_safe_id(video_id)
        out_path = out_dir / f"{safe_id}.npy"

        if out_path.exists():
            skipped += 1
            continue

        label = float(row["label"]) if "label" in df.columns else 0.0
        method = str(row["method"]) if "method" in df.columns else ""
        comp = str(row["compression"]) if "compression" in df.columns else ""
        vpath = str(row["video_path"]) if "video_path" in df.columns else ""

        feat = [label]

        # one-hot method
        for m in method_vocab:
            feat.append(1.0 if method == m else 0.0)

        # one-hot compression
        for c in comp_vocab:
            feat.append(1.0 if comp == c else 0.0)

        # simple deterministic numeric features
        feat.append(float(len(vpath)))
        feat.append(float(len(Path(vpath).name)))

        np.save(out_path, np.array(feat, dtype=np.float32))
        saved += 1

    print("\nDone.")
    print("Saved:", saved)
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()
