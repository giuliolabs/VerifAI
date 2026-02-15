"""
Build FaceForensics++_C23 manifest + train/val/test splits
=========================================================

Builds a dataset manifest and reproducible train/val/test splits for the
FaceForensics++ dataset in the specific "flat folder" layout used in this repo:

data/raw/FaceForensics++_C23/
  original/*.mp4                         -> real (label 0)
  Deepfakes/*.mp4                        -> fake (label 1)
  Face2Face/*.mp4                        -> fake (label 1)
  FaceSwap/*.mp4                         -> fake (label 1)
  NeuralTextures/*.mp4                   -> fake (label 1)
  DeepFakeDetection/*.mp4 (if present)   -> fake (label 1)
  FaceShifter/*.mp4 (if present)         -> fake (label 1)
  csv/*.csv                              -> ignored

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

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

from pathlib import Path
import random
import pandas as pd
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

RAW_ROOT = Path("data/raw/FaceForensics++_C23")
COMPRESSION = "c23"

REAL_FOLDER = "original"

# All method folders present in YOUR tree (safe if some are missing)
FAKE_FOLDERS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "DeepFakeDetection",
    "FaceShifter",
]

IGNORE_FOLDERS = {"csv"}


def list_videos(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def make_video_id(dataset: str, path: Path, raw_root: Path) -> str:
    rel = path.relative_to(raw_root).as_posix()
    safe = rel.replace("/", "__").replace(":", "")
    return f"{dataset}__{safe}"


def split_ids(items: list[str], seed: int = 42):
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
    dataset = "FaceForensics++"

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT not found: {RAW_ROOT.resolve()}")

    rows = []

    # ---- REAL ----
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
    for method in FAKE_FOLDERS:
        method_dir = RAW_ROOT / method
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

    df = pd.DataFrame(rows)

    out_manifest = Path("data/metadata/faceforensics++_c23_manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print("\nSaved manifest:", out_manifest)
    print("Total rows:", len(df))
    print("Label distribution:\n", df["label"].value_counts(dropna=False))
    print("\nMethod distribution:\n", df["method"].value_counts(dropna=False))

    # Train/val/test split (GROUPED to avoid leakage across methods)
    group_ids = sorted(df["group_id"].unique().tolist())
    train_g, val_g, test_g = split_ids(group_ids, seed=42)

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    def save_split(split_name: str, group_list: list[str]):
        split_df = df[df["group_id"].isin(group_list)].copy()
        out_path = Path("data/splits") / f"faceforensics++_c23_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Saved split: {out_path} rows={len(split_df)}")

    save_split("train", train_g)
    save_split("val", val_g)
    save_split("test", test_g)

if __name__ == "__main__":
    main()
