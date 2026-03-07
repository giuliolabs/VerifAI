"""
Build FakeAVCeleb_v1.2 manifest + train/val/test splits
=======================================================

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

------------------------------------------------
NOTES
------------------------------------------------
- Splits are currently created at video level.
- This is sufficient for the current pipeline and relabeling fix.
- A later improvement would be subject-level splitting to reduce identity leakage.

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


def split_ids(items, seed=42):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)

    total = len(items)
    train_end = int(0.8 * total)
    val_end = train_end + int(0.1 * total)

    train_ids = items[:train_end]
    val_ids = items[train_end:val_end]
    test_ids = items[val_end:]

    return train_ids, val_ids, test_ids


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

    out_manifest = Path("data/metadata/fakeavceleb_manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print("\nSaved manifest:", out_manifest)
    print("Total rows:", len(df))
    print("Label distribution:\n", df["label"].value_counts(dropna=False))
    print("\nAV category distribution:\n", df["av_category"].value_counts(dropna=False))

    ids = df["video_id"].tolist()
    train_ids, val_ids, test_ids = split_ids(ids, seed=42)

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    def save_split(split_name, id_list):
        split_df = df[df["video_id"].isin(id_list)].copy()
        out_path = Path("data/splits") / f"fakeavceleb_{split_name}.csv"
        split_df.to_csv(out_path, index=False)

        print(f"\nSaved split: {out_path}  rows={len(split_df)}")
        print(f"{split_name} label distribution:")
        print(split_df["label"].value_counts(dropna=False))
        print(f"\n{split_name} AV category distribution:")
        print(split_df["av_category"].value_counts(dropna=False))

    save_split("train", train_ids)
    save_split("val", val_ids)
    save_split("test", test_ids)


if __name__ == "__main__":
    main()
