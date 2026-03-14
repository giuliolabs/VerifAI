"""
Build DeeperForensics Manifest + Train/Val/Test Splits
=====================================================

This script scans the DeeperForensics dataset, builds a clean manifest of
all available videos, and then creates reproducible train, validation,
and test split CSV files.

Expected dataset layout:

    data/raw/DeeperForensics/
        real/...
        fake/
            manipulated_videos_1/
                end_to_end_random_level/*.mp4
                reenact_postprocess/*.mp4

The script works at video level, meaning each full video belongs to
only one split. This is important because splitting at frame level would
cause data leakage and make evaluation less reliable.

Outputs created:

    data/metadata/deeperforensics_manifest.csv
    data/splits/deeperforensics_train.csv
    data/splits/deeperforensics_val.csv
    data/splits/deeperforensics_test.csv

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Path is used for clean and platform-independent file/folder handling
from pathlib import Path

# pandas stores the manifest and split tables
import pandas as pd

# tqdm adds progress bars while scanning the dataset
from tqdm import tqdm

# random is used to shuffle IDs reproducibly before splitting
import random


# Supported video extensions that should be included in the manifest
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_videos(root: Path):
    """
    Recursively collect all video files under a folder.

    If the folder does not exist, return an empty list instead of failing.
    This keeps the script more robust if a dataset path is incomplete.
    """
    if not root.exists():
        return []

    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def make_video_id(dataset: str, path: Path, raw_root: Path):
    """
    Build a stable video ID from the file's relative path.

    The relative path is converted into a safer string by replacing
    slashes with double underscores. This helps keep IDs unique and
    makes them easier to use later in CSV files and preprocessing steps.
    """
    rel = path.relative_to(raw_root).as_posix()
    safe = rel.replace("/", "__").replace(":", "")
    return f"{dataset}__{safe}"


def split_ids(items, seed=42):
    """
    Split a list of video IDs into train / val / test sets.

    Ratios used:
    - 80% train
    - 10% val
    - 10% test

    A fixed seed is used so the split stays reproducible.
    """
    rng = random.Random(seed)

    # Shuffle in place so the split is random but repeatable
    rng.shuffle(items)

    n = len(items)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    return train, val, test


def main():
    """
    Main workflow:
    1. Locate real and fake videos
    2. Build the dataset manifest
    3. Save the manifest CSV
    4. Create train / val / test splits
    5. Save split CSV files
    """
    dataset = "DeeperForensics"
    raw_root = Path("data/raw/DeeperForensics")

    # Expected real/fake top-level folders
    real_root = raw_root / "real"
    fake_root = raw_root / "fake"

    # Collect all real and fake videos separately
    real_videos = list_videos(real_root)
    fake_videos = list_videos(fake_root)

    rows = []

    # Add all real videos to the manifest with label 0
    for vp in tqdm(real_videos, desc="Scanning DeeperForensics REAL"):
        rows.append({
            "dataset": dataset,
            "video_id": make_video_id(dataset, vp, raw_root),
            "video_path": str(vp),
            "label": 0
        })

    # Add all fake videos to the manifest with label 1
    for vp in tqdm(fake_videos, desc="Scanning DeeperForensics FAKE"):
        rows.append({
            "dataset": dataset,
            "video_id": make_video_id(dataset, vp, raw_root),
            "video_path": str(vp),
            "label": 1
        })

    # Convert collected rows into a DataFrame for easier saving and filtering
    df = pd.DataFrame(rows)

    # Save the full manifest before splitting
    out_manifest = Path("data/metadata/deeperforensics_manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print("\nSaved manifest:", out_manifest)
    print("Total rows:", len(df))
    print("Label distribution:\n", df["label"].value_counts(dropna=False))

    # Create train / val / test splits at VIDEO LEVEL
    # This avoids leakage because each video_id goes into only one split
    ids = df["video_id"].tolist()
    train_ids, val_ids, test_ids = split_ids(ids, seed=42)

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    def save_split(split_name, id_list):
        """
        Save one split CSV by keeping only rows whose video_id
        belongs to the given ID list.
        """
        split_df = df[df["video_id"].isin(id_list)].copy()
        out_path = Path("data/splits") / f"deeperforensics_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Saved split: {out_path}  rows={len(split_df)}")

    # Write all three split files
    save_split("train", train_ids)
    save_split("val", val_ids)
    save_split("test", test_ids)


if __name__ == "__main__":
    main()