"""
Build FaceForensics++_C23 Manifest + Train/Val/Test Splits
=========================================================

This script builds a clean dataset manifest and reproducible train,
validation, and test splits for the FaceForensics++ C23 dataset, using
the flat folder structure already used in this project.

Expected dataset layout:

data/raw/FaceForensics++_C23/
  original/*.mp4                         -> real (label 0)
  Deepfakes/*.mp4                        -> fake (label 1)
  Face2Face/*.mp4                        -> fake (label 1)
  FaceSwap/*.mp4                         -> fake (label 1)
  NeuralTextures/*.mp4                   -> fake (label 1)
  DeepFakeDetection/*.mp4 (if present)   -> fake (label 1)
  FaceShifter/*.mp4 (if present)         -> fake (label 1)
  csv/*.csv                              -> ignored

The script:
- scans all supported video folders
- builds one manifest row per video
- infers labels from folder names
- creates grouped train/val/test splits
- saves everything as CSV files for later preprocessing and training

The split is done using `group_id` so related samples stay together and
data leakage is reduced.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
    pip install pandas tqdm

------------------------------------------------
RUN
------------------------------------------------
python pipelines/splits/build_faceforensics++_c23_manifest.py

------------------------------------------------
OUTPUTS
------------------------------------------------
data/metadata/faceforensics++_c23_manifest.csv
data/splits/faceforensics++_c23_train.csv
data/splits/faceforensics++_c23_val.csv
data/splits/faceforensics++_c23_test.csv

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- Video-level splitting is seeded for reproducibility.
- Manifest standardizes dataset access for VerifAI pipelines.
- Uses folder name to infer label (original=real, others=fake).

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Path is used for clean cross-platform file handling
from pathlib import Path

# random is used for reproducible splitting
import random

# pandas stores the manifest and split tables
import pandas as pd

# tqdm adds progress bars while scanning videos
from tqdm import tqdm


# Supported video file extensions
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Root folder of the FaceForensics++ C23 dataset in this project
RAW_ROOT = Path("data/raw/FaceForensics++_C23")

# This tag is stored in the manifest so it is clear which compression setting was used
COMPRESSION = "c23"

# Folder containing real videos
REAL_FOLDER = "original"

# All fake-method folders that may exist in this dataset layout
# The script is safe even if some of them are missing
FAKE_FOLDERS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "DeepFakeDetection",
    "FaceShifter",
]

# Folders that should be ignored during scanning
IGNORE_FOLDERS = {"csv"}


def list_videos(folder: Path) -> list[Path]:
    """
    Return all video files found under a folder, recursively.

        If the folder does not exist, return an empty list instead of failing.
    """
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def make_video_id(dataset: str, path: Path, raw_root: Path) -> str:
    """
    Build a stable video_id using the relative file path.

    Slashes are replaced so the ID is easy to store in CSV files
    and remains unique inside the dataset.
    """
    rel = path.relative_to(raw_root).as_posix()
    safe = rel.replace("/", "__").replace(":", "")
    return f"{dataset}__{safe}"


def split_ids(items: list[str], seed: int = 42):
    """
    Split a list of IDs into train / val / test groups.

    Ratios used:
    - 80% train
    - 10% val
    - 10% test

    The seed makes the split reproducible.
    """
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)

    n = len(items)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    return train, val, test


def main() -> None:
    """
    Main workflow:
    1. Check dataset root exists
    2. Scan real and fake folders
    3. Build manifest
    4. Save manifest
    5. Create grouped train/val/test splits
    6. Save split CSV files
    """
    dataset = "FaceForensics++"

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT not found: {RAW_ROOT.resolve()}")

    rows = []

    # ---- REAL ----
    # Scan the 'original' folder and mark all those videos as real
    real_dir = RAW_ROOT / REAL_FOLDER
    real_videos = list_videos(real_dir)

    for vp in tqdm(real_videos, desc=f"Scanning FF++ REAL ({REAL_FOLDER})"):
        rows.append({
            "dataset": dataset,
            "video_id": make_video_id(dataset, vp, RAW_ROOT),
            "video_path": str(vp),
            "label": 0,
            "method": REAL_FOLDER,
            "compression": COMPRESSION,
            "group_id": vp.stem,
        })

    # ---- FAKE ----
    # Scan each fake manipulation folder and label all videos as fake
    for method in FAKE_FOLDERS:
        method_dir = RAW_ROOT / method

        # Extra safety: skip ignored folders if they somehow appear here
        if method_dir.name in IGNORE_FOLDERS:
            continue

        method_videos = list_videos(method_dir)

        for vp in tqdm(method_videos, desc=f"Scanning FF++ FAKE ({method})"):
            rows.append({
                "dataset": dataset,
                "video_id": make_video_id(dataset, vp, RAW_ROOT),
                "video_path": str(vp),
                "label": 1,
                "method": method,
                "compression": COMPRESSION,
                "group_id": vp.stem,
            })

    # Convert collected rows into a DataFrame for saving and splitting
    df = pd.DataFrame(rows)

    # Save the full manifest first so it can be reused by other scripts
    out_manifest = Path("data/metadata/faceforensics++_c23_manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print("\nSaved manifest:", out_manifest)
    print("Total rows:", len(df))
    print("Label distribution:\n", df["label"].value_counts(dropna=False))
    print("\nMethod distribution:\n", df["method"].value_counts(dropna=False))

    # Build grouped splits using group_id rather than splitting rows directly
    # This helps reduce leakage by keeping related samples together
    group_ids = sorted(df["group_id"].unique().tolist())
    train_g, val_g, test_g = split_ids(group_ids, seed=42)

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    def save_split(split_name: str, group_list: list[str]):
        """
        Save one split CSV by filtering rows whose group_id belongs to that split.
        """
        split_df = df[df["group_id"].isin(group_list)].copy()
        out_path = Path("data/splits") / f"faceforensics++_c23_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Saved split: {out_path} rows={len(split_df)}")

    # Write all three split files
    save_split("train", train_g)
    save_split("val", val_g)
    save_split("test", test_g)


if __name__ == "__main__":
    main()