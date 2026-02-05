"""
Build DeeperForensics manifest + train/val/test splits

Dataset layout (as in your folder tree):
  data/raw/DeeperForensics/
    real/...
    fake/
      manipulated_videos_1/
        end_to_end_random_level/*.mp4
        reenact_postprocess/*.mp4

Dependencies:
    pip install opencv-python pandas tqdm numpy pillow
    pip install librosa soundfile
    pip install pandas tqdm

Run:
    python pipelines/splits/build_deeperforensics_manifest.py

Outputs:
    data/metadata/deeperforensics_manifest.csv
    data/splits/deeperforensics_train.csv
    data/splits/deeperforensics_val.csv
    data/splits/deeperforensics_test.csv
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import random

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_videos(root: Path):
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


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


def main():
    dataset = "DeeperForensics"
    raw_root = Path("data/raw/DeeperForensics")

    real_root = raw_root / "real"
    fake_root = raw_root / "fake"

    real_videos = list_videos(real_root)
    fake_videos = list_videos(fake_root)

    rows = []

    for vp in tqdm(real_videos, desc="Scanning DeeperForensics REAL"):
        rows.append({
            "dataset": dataset,
            "video_id": make_video_id(dataset, vp, raw_root),
            "video_path": str(vp),
            "label": 0
        })

    for vp in tqdm(fake_videos, desc="Scanning DeeperForensics FAKE"):
        rows.append({
            "dataset": dataset,
            "video_id": make_video_id(dataset, vp, raw_root),
            "video_path": str(vp),
            "label": 1
        })

    df = pd.DataFrame(rows)

    out_manifest = Path("data/metadata/deeperforensics_manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print("\nSaved manifest:", out_manifest)
    print("Total rows:", len(df))
    print("Label distribution:\n", df["label"].value_counts(dropna=False))

    # Create train/val/test splits at VIDEO LEVEL
    ids = df["video_id"].tolist()
    train_ids, val_ids, test_ids = split_ids(ids, seed=42)

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    def save_split(split_name, id_list):
        split_df = df[df["video_id"].isin(id_list)].copy()
        out_path = Path("data/splits") / f"deeperforensics_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Saved split: {out_path}  rows={len(split_df)}")

    save_split("train", train_ids)
    save_split("val", val_ids)
    save_split("test", test_ids)


if __name__ == "__main__":
    main()
