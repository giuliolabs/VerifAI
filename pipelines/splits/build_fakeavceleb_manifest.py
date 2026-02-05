"""
Build FakeAVCeleb_v1.2 manifest + train/val/test splits

Dataset layout (as in your folder tree):
  data/raw/FakeAVCeleb_v1.2/
    FakeVideo-FakeAudio/<Ethnicity>/<gender>/<subject_id>/*.mp4
    FakeVideo-RealAudio/<Ethnicity>/<gender>/<subject_id>/*.mp4
    RealVideo-FakeAudio/<Ethnicity>/<gender>/<subject_id>/*.mp4
    RealVideo-RealAudio/<Ethnicity>/<gender>/<subject_id>/*.mp4
    meta_data.csv

Dependencies:
    pip install pandas tqdm

Run:
    python pipelines/splits/build_fakeavceleb_manifest.py

Outputs:
    data/metadata/fakeavceleb_manifest.csv
    data/splits/fakeavceleb_train.csv
    data/splits/fakeavceleb_val.csv
    data/splits/fakeavceleb_test.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import random

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

AV_CATEGORIES = {
    "FakeVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "RealVideo-FakeAudio",
    "RealVideo-RealAudio"
}

def list_videos(raw_root: Path):
    videos = []
    for cat in AV_CATEGORIES:
        cat_dir = raw_root / cat
        if not cat_dir.exists():
            continue
        for p in cat_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                videos.append(p)
    return videos

def make_video_id(dataset: str, path: Path, raw_root: Path):
    rel = path.relative_to(raw_root).as_posix()
    safe = rel.replace("/", "__").replace(":", "")
    return f"{dataset}__{safe}"

def split_ids(items, seed=42):
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test

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

    # label rule: FakeVideo-* => fake(1), RealVideo-* => real(0)
    label = 1 if av_category.startswith("FakeVideo") else 0

    return av_category, ethnicity, gender, subject_id, label

def main():
    dataset = "FakeAVCeleb_v1.2"
    raw_root = Path("data/raw/FakeAVCeleb_v1.2")

    videos = list_videos(raw_root)
    rows = []

    for vp in tqdm(videos, desc="Scanning FakeAVCeleb_v1.2"):
        av_category, ethnicity, gender, subject_id, label = parse_fakeavceleb_fields(vp, raw_root)
        rows.append({
            "dataset": dataset,
            "video_id": make_video_id(dataset, vp, raw_root),
            "video_path": str(vp),
            "label": label,
            "av_category": av_category,
            "ethnicity": ethnicity,
            "gender": gender,
            "subject_id": subject_id
        })

    df = pd.DataFrame(rows)

    out_manifest = Path("data/metadata/fakeavceleb_manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print("\nSaved manifest:", out_manifest)
    print("Total rows:", len(df))
    print("Label distribution:\n", df["label"].value_counts(dropna=False))
    print("\nAV category distribution:\n", df["av_category"].value_counts(dropna=False).head(20))

    # Create train/val/test splits at VIDEO LEVEL
    ids = df["video_id"].tolist()
    train_ids, val_ids, test_ids = split_ids(ids, seed=42)

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    def save_split(split_name, id_list):
        split_df = df[df["video_id"].isin(id_list)].copy()
        out_path = Path("data/splits") / f"fakeavceleb_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Saved split: {out_path}  rows={len(split_df)}")

    save_split("train", train_ids)
    save_split("val", val_ids)
    save_split("test", test_ids)

if __name__ == "__main__":
    main()
