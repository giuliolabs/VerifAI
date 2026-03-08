"""
Build FakeAVCeleb_v1.2 manifest + subject-level train/val/test splits
=====================================================================

Dataset layout (as in your folder tree):
  data/raw/FakeAVCeleb_v1.2/
    FakeVideo-FakeAudio/<Ethnicity>/<gender>/<subject_id>/*.mp4
    FakeVideo-RealAudio/<Ethnicity>/<gender>/<subject_id>/*.mp4
    RealVideo-FakeAudio/<Ethnicity>/<gender>/<subject_id>/*.mp4
    RealVideo-RealAudio/<Ethnicity>/<gender>/<subject_id>/*.mp4
    meta_data.csv

------------------------------------------------
LABEL DEFINITION
------------------------------------------------
This version builds labels for overall clip authenticity:

    RealVideo-RealAudio  -> 0 (real)
    FakeVideo-RealAudio  -> 1 (fake)
    RealVideo-FakeAudio  -> 1 (fake)
    FakeVideo-FakeAudio  -> 1 (fake)

So the model learns whether the whole clip is fake in any modality.

------------------------------------------------
SPLITTING STRATEGY
------------------------------------------------
This version performs SUBJECT-LEVEL splitting:
- each subject_id appears in only one split
- reduces identity leakage across train/val/test
- provides a more realistic evaluation

------------------------------------------------
DEPENDENCIES
------------------------------------------------
    pip install pandas tqdm

------------------------------------------------
RUN
------------------------------------------------
    python pipelines/splits/build_fakeavceleb_manifest.py

------------------------------------------------
OUTPUTS
------------------------------------------------
    data/metadata/fakeavceleb_manifest.csv
    data/splits/fakeavceleb_train.csv
    data/splits/fakeavceleb_val.csv
    data/splits/fakeavceleb_test.csv

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from pathlib import Path
import random

import pandas as pd
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

AV_CATEGORIES = {
    "FakeVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "RealVideo-FakeAudio",
    "RealVideo-RealAudio",
}


def list_videos(raw_root: Path):
    videos = []

    for category in AV_CATEGORIES:
        category_dir = raw_root / category
        if not category_dir.exists():
            continue

        for path in category_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                videos.append(path)

    return videos


def make_video_id(dataset: str, path: Path, raw_root: Path):
    rel = path.relative_to(raw_root).as_posix()
    safe = rel.replace("/", "__").replace(":", "")
    return f"{dataset}__{safe}"


def split_subject_ids(subject_ids, seed=42):
    rng = random.Random(seed)
    subject_ids = list(subject_ids)
    rng.shuffle(subject_ids)

    total = len(subject_ids)
    train_end = int(0.8 * total)
    val_end = train_end + int(0.1 * total)

    train_subjects = set(subject_ids[:train_end])
    val_subjects = set(subject_ids[train_end:val_end])
    test_subjects = set(subject_ids[val_end:])

    return train_subjects, val_subjects, test_subjects


def parse_fakeavceleb_fields(video_path: Path, raw_root: Path):
    """
    Expected relative path:
      <AV_CATEGORY>/<Ethnicity>/<gender>/<subject_id>/<filename>.mp4
    """
    rel_parts = video_path.relative_to(raw_root).parts

    av_category = rel_parts[0] if len(rel_parts) > 0 else ""
    ethnicity = rel_parts[1] if len(rel_parts) > 1 else ""
    gender = rel_parts[2] if len(rel_parts) > 2 else ""
    subject_id = rel_parts[3] if len(rel_parts) > 3 else ""

    # Clip-level authenticity rule:
    # Only RealVideo-RealAudio is real.
    label = 0 if av_category == "RealVideo-RealAudio" else 1

    return av_category, ethnicity, gender, subject_id, label


def main():
    dataset = "FakeAVCeleb_v1.2"
    raw_root = Path("data/raw/FakeAVCeleb_v1.2")

    if not raw_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {raw_root}")

    videos = list_videos(raw_root)
    rows = []

    for video_path in tqdm(videos, desc="Scanning FakeAVCeleb_v1.2"):
        av_category, ethnicity, gender, subject_id, label = parse_fakeavceleb_fields(video_path, raw_root)

        rows.append({
            "dataset": dataset,
            "video_id": make_video_id(dataset, video_path, raw_root),
            "video_path": str(video_path),
            "label": label,
            "av_category": av_category,
            "ethnicity": ethnicity,
            "gender": gender,
            "subject_id": subject_id,
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No videos found. Check the FakeAVCeleb dataset path.")

    out_manifest = Path("data/metadata/fakeavceleb_manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print("\nSaved manifest:", out_manifest)
    print("Total rows:", len(df))
    print("Unique subjects:", df["subject_id"].nunique())
    print("Label distribution:\n", df["label"].value_counts(dropna=False))
    print("\nAV category distribution:\n", df["av_category"].value_counts(dropna=False))

    unique_subjects = sorted(df["subject_id"].dropna().astype(str).unique().tolist())
    train_subjects, val_subjects, test_subjects = split_subject_ids(unique_subjects, seed=42)

    train_df = df[df["subject_id"].astype(str).isin(train_subjects)].copy()
    val_df = df[df["subject_id"].astype(str).isin(val_subjects)].copy()
    test_df = df[df["subject_id"].astype(str).isin(test_subjects)].copy()

    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "fakeavceleb_train.csv"
    val_path = splits_dir / "fakeavceleb_val.csv"
    test_path = splits_dir / "fakeavceleb_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    def print_split_stats(name: str, split_df: pd.DataFrame):
        print(f"\n{name} split")
        print("Rows:", len(split_df))
        print("Unique subjects:", split_df['subject_id'].nunique())
        print("Label distribution:")
        print(split_df["label"].value_counts(dropna=False))
        print("\nAV category distribution:")
        print(split_df["av_category"].value_counts(dropna=False))

    print("\nSaved split files:")
    print(train_path)
    print(val_path)
    print(test_path)

    print_split_stats("Train", train_df)
    print_split_stats("Val", val_df)
    print_split_stats("Test", test_df)

    overlap_train_val = train_subjects.intersection(val_subjects)
    overlap_train_test = train_subjects.intersection(test_subjects)
    overlap_val_test = val_subjects.intersection(test_subjects)

    print("\nSubject overlap checks:")
    print("Train ∩ Val :", len(overlap_train_val))
    print("Train ∩ Test:", len(overlap_train_test))
    print("Val ∩ Test  :", len(overlap_val_test))


if __name__ == "__main__":
    main()
