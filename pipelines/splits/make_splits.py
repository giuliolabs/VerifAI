"""
Create Train / Validation / Test Splits for FaceForensics++ C23
===============================================================

This script builds the train, validation, and test CSV split files
for the FaceForensics++ C23 dataset.

The split is performed at **video level**, not frame level, which is
important because all frames from the same video must stay in the same
subset. This avoids data leakage between training and evaluation.

The script:
- scans the dataset root for video files
- infers whether each video is real or fake from its folder path
- shuffles the dataset in a reproducible way
- creates 70% / 15% / 15% train, val, test splits
- saves the result as CSV files in `data/splits`

Output files:
    data/splits/faceforensics++_c23_train.csv
    data/splits/faceforensics++_c23_val.csv
    data/splits/faceforensics++_c23_test.csv

Each CSV contains:
    video_path,label,video_id

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Path is used for dataset folder traversal in a platform-independent way
from pathlib import Path

# pandas is used to store and save the collected video records as CSV files
import pandas as pd


# Valid video file extensions to include when scanning the dataset
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# Folder names that indicate manipulated / fake videos in FaceForensics++
FAKE_FOLDERS = {
    "deepfakes",
    "face2face",
    "faceswap",
    "faceshifter",
    "neuraltextures",
}


def infer_label(video_path: Path) -> int:
    """
    Infer the binary label from the video path.

    Label rules:
    - 0 = real
    - 1 = fake
    - -1 = unknown / unsupported path

    The function checks whether the path contains:
    - 'original' -> real
    - one of the known fake manipulation folder names -> fake
    """
    parts = [p.lower() for p in video_path.parts]

    # Original videos are treated as real
    if "original" in parts:
        return 0

    # If any known manipulation folder is present, mark as fake
    for name in FAKE_FOLDERS:
        if name in parts:
            return 1

    # If the path does not match either rule, label is unknown
    return -1


def collect_videos(root: Path):
    """
    Recursively collect all valid video files under the dataset root.

    For each video, create a record containing:
    - full normalized path
    - inferred label
    - video_id
    """
    records = []

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            label = infer_label(p)

            records.append({
                "video_path": str(p).replace("\\", "/"),
                "label": label,
                "video_id": p.stem
            })

    return records


def main():
    """
    Main execution logic.

    This function:
    - checks the dataset root exists
    - collects video records
    - removes unknown-label entries
    - shuffles records reproducibly
    - splits them into train / val / test sets
    - writes the CSV outputs
    """
    ffpp_root = Path("data/raw/FaceForensics++_C23")

    if not ffpp_root.exists():
        raise FileNotFoundError(ffpp_root)

    # Scan dataset and collect video metadata
    records = collect_videos(ffpp_root)
    df = pd.DataFrame(records)

    # Remove videos whose label could not be inferred
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    print("Total FF++ videos:", len(df))
    print("Label distribution:")
    print(df["label"].value_counts())

    # Shuffle rows in a reproducible way
    # random_state=42 ensures the same split every time
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Define split sizes
    n = len(df)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    # Slice the shuffled dataframe into train / val / test subsets
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    # Create output directory if needed
    out_dir = Path("data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV files
    train_df.to_csv(out_dir / "faceforensics++_c23_train.csv", index=False)
    val_df.to_csv(out_dir / "faceforensics++_c23_val.csv", index=False)
    test_df.to_csv(out_dir / "faceforensics++_c23_test.csv", index=False)

    print("Splits written:")
    print(" train:", len(train_df))
    print(" val:", len(val_df))
    print(" test:", len(test_df))


if __name__ == "__main__":
    main()