"""
Build Celeb-DF v2 Manifest + Train/Val/Test Splits (Protocol-Aware)
===================================================================

This script builds a full manifest for the Celeb-DF v2 dataset and then
creates train, validation, and test split CSV files in a way that follows
the official protocol as closely as possible.

Dataset layout (common):
  data/raw/Celeb-DF-v2/
    Celeb-real/
      videos/*.mp4           (or sometimes *.mp4 directly)
    YouTube-real/
      videos/*.mp4           (or sometimes *.mp4 directly)
    Celeb-synthesis/
      videos/*.mp4           (or sometimes *.mp4 directly)
    List_of_testing_videos.txt

Important idea:
- Celeb-DF v2 provides an official test list in `List_of_testing_videos.txt`
- This script uses that official list as the TEST split
- All remaining videos are split into TRAIN and VAL (default 90/10)
- This keeps evaluation more aligned with the published dataset protocol

Outputs:
    data/metadata/celebdfv2_manifest.csv
    data/splits/celebdfv2_train.csv
    data/splits/celebdfv2_val.csv
    data/splits/celebdfv2_test.csv

Each CSV stores standardized information such as:
- dataset name
- video_id
- video_path
- binary label
- source folder
- whether the sample belongs to the official protocol test list

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Path is used for safe and readable file/folder handling
from pathlib import Path

# pandas is used to store manifest rows and save split CSV files
import pandas as pd

# tqdm adds a progress bar while scanning the dataset
from tqdm import tqdm

# random is used for reproducible train/validation splitting
import random


# Supported video extensions
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_videos(root: Path):
    """
    Recursively collect all video files under a folder.

    If the folder does not exist, return an empty list instead of failing.
    """
    if not root.exists():
        return []

    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def dataset_video_roots(raw_root: Path):
    """
    Return all possible roots that may contain Celeb-DF v2 videos.

    This handles two common layouts:

    1) Videos stored inside:
         <split>/videos/*.mp4

    2) Videos stored directly inside:
         <split>/*.mp4

    By including both possibilities, the script becomes more robust
    across slightly different dataset extractions.
    """
    roots = []

    for name in ["Celeb-real", "YouTube-real", "Celeb-synthesis"]:
        base = raw_root / name

        if not base.exists():
            continue

        videos_dir = base / "videos"

        if videos_dir.exists():
            roots.append(videos_dir)

        # Also include the base folder itself in case videos are stored directly there
        roots.append(base)

    return roots


def make_video_id(dataset: str, path: Path, raw_root: Path):
    """
    Build a stable video_id using the file's relative path.

    Slashes are replaced with double underscores so the ID is easy
    to store and stays unique across the dataset.
    """
    rel = path.relative_to(raw_root).as_posix()
    safe = rel.replace("/", "__").replace(":", "")
    return f"{dataset}__{safe}"


def read_test_list(raw_root: Path):
    """
    Read the official Celeb-DF v2 test list if it exists.

    The file often contains lines like:
      YouTube-real/00238.mp4
      Celeb-real/00001.mp4
      Celeb-synthesis/00001.mp4

    But sometimes the actual extracted dataset stores files under:
      <split>/videos/*.mp4

    So this function prepares two sets for matching:
    1) exact relative paths from the file
    2) versions with 'videos/' inserted

    Returns:
        test_rel              -> direct relative paths
        test_rel_with_videos  -> same paths but with 'videos/' inserted
    """
    test_file = raw_root / "List_of_testing_videos.txt"

    if not test_file.exists():
        return set(), set()

    lines = []

    for line in test_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove leading "./" or "/" so matching is more stable
        while line.startswith("./"):
            line = line[2:]

        while line.startswith("/"):
            line = line[1:]

        lines.append(line)

    test_rel = set()
    test_rel_with_videos = set()

    for rel in lines:
        rel = rel.replace("\\", "/")
        test_rel.add(rel)

        parts = rel.split("/")

        # Insert "videos" after the top folder when possible
        if len(parts) >= 2:
            test_rel_with_videos.add("/".join([parts[0], "videos"] + parts[1:]))

    return test_rel, test_rel_with_videos


def infer_label_and_source(video_path: Path, raw_root: Path):
    """
    Infer the binary label and source category from the video path.

    Label rule:
    - Celeb-synthesis -> fake (1)
    - Celeb-real      -> real (0)
    - YouTube-real    -> real (0)

    Returns:
        label, source_folder_name
    """
    rel_parts = video_path.relative_to(raw_root).parts
    top = rel_parts[0] if len(rel_parts) > 0 else ""

    if top == "Celeb-synthesis":
        return 1, top

    if top in {"Celeb-real", "YouTube-real"}:
        return 0, top

    # Fallback rule in case folder names differ slightly
    return (1 if "synth" in top.lower() else 0), top


def split_train_val(items, seed=42, val_ratio=0.1):
    """
    Split a list of video IDs into train and validation sets.

    Default:
    - 90% train
    - 10% val

    A fixed seed is used to make the split reproducible.
    """
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_val = int(val_ratio * n)

    val = items[:n_val]
    train = items[n_val:]

    return train, val


def main():
    """
    Main execution flow.

    This function:
    1. Locates all Celeb-DF v2 videos
    2. Reads the official protocol test list
    3. Builds the manifest
    4. Saves the manifest CSV
    5. Creates protocol-aware train / val / test splits
    6. Saves split CSV files
    """
    dataset = "CelebDFv2"
    raw_root = Path("data/raw/Celeb-DF-v2")

    # Gather all possible roots that may contain videos
    roots = dataset_video_roots(raw_root)

    videos = []
    seen = set()

    for r in roots:
        for p in list_videos(r):
            # Avoid duplicates if the same file is found through two paths
            if p in seen:
                continue

            seen.add(p)

            # Extra safety: only keep files actually under raw_root
            try:
                _ = p.relative_to(raw_root)
            except ValueError:
                continue

            videos.append(p)

    # Read the official protocol test list
    test_rel, test_rel_with_videos = read_test_list(raw_root)

    rows = []

    for vp in tqdm(videos, desc="Scanning Celeb-DF v2"):
        label, source = infer_label_and_source(vp, raw_root)
        rel = vp.relative_to(raw_root).as_posix()

        # Some files may exist as ".../videos/xxx.mp4" in the dataset,
        # while the official list may only say ".../xxx.mp4"
        rel_no_videos = rel.replace("/videos/", "/")

        # A sample is protocol test if it matches in any supported form
        is_test = (
            (rel in test_rel) or
            (rel in test_rel_with_videos) or
            (rel_no_videos in test_rel)
        )

        rows.append({
            "dataset": dataset,
            "video_id": make_video_id(dataset, vp, raw_root),
            "video_path": str(vp),
            "label": label,
            "source": source,
            "is_protocol_test": int(is_test),
        })

    df = pd.DataFrame(rows)

    # Save the full manifest before creating split subsets
    out_manifest = Path("data/metadata/celebdfv2_manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print("\nSaved manifest:", out_manifest)
    print("Total rows:", len(df))
    print("Label distribution:\n", df["label"].value_counts(dropna=False))
    print("Source distribution:\n", df["source"].value_counts(dropna=False))
    print("Protocol TEST count:", int(df["is_protocol_test"].sum()))

    # Build final splits:
    # - TEST comes from the official protocol list
    # - Remaining videos are split into TRAIN and VAL
    test_df = df[df["is_protocol_test"] == 1].copy()
    rest_df = df[df["is_protocol_test"] == 0].copy()

    rest_ids = rest_df["video_id"].tolist()
    train_ids, val_ids = split_train_val(rest_ids, seed=42, val_ratio=0.1)

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    def save_split(split_name, split_df):
        """
        Save one split DataFrame to CSV.
        """
        out_path = Path("data/splits") / f"celebdfv2_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Saved split: {out_path}  rows={len(split_df)}")

    # Save protocol-aware split files
    save_split("test", test_df)
    save_split("train", rest_df[rest_df["video_id"].isin(train_ids)].copy())
    save_split("val", rest_df[rest_df["video_id"].isin(val_ids)].copy())


if __name__ == "__main__":
    main()