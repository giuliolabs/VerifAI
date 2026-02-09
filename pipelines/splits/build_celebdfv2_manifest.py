"""
Build Celeb-DF v2 manifest + train/val/test splits (protocol-aware)

Dataset layout (common):
  data/raw/Celeb-DF-v2/
    Celeb-real/
      videos/*.mp4           (or sometimes *.mp4 directly)
    YouTube-real/
      videos/*.mp4           (or sometimes *.mp4 directly)
    Celeb-synthesis/
      videos/*.mp4           (or sometimes *.mp4 directly)
    List_of_testing_videos.txt

Notes:
- Celeb-DF v2 provides an official test list in List_of_testing_videos.txt.
  This script uses that list as TEST.
- The remaining videos are split into TRAIN/VAL (default 90/10) with a seed.

Dependencies:
    pip install pandas tqdm

Run:
    python pipelines/splits/build_celebdfv2_manifest.py

Outputs:
    data/metadata/celebdfv2_manifest.csv
    data/splits/celebdfv2_train.csv
    data/splits/celebdfv2_val.csv
    data/splits/celebdfv2_test.csv
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


def dataset_video_roots(raw_root: Path):
    """
    Handle both:
      <split>/videos/*.mp4
    and:
      <split>/*.mp4
    """
    roots = []
    for name in ["Celeb-real", "YouTube-real", "Celeb-synthesis"]:
        base = raw_root / name
        if not base.exists():
            continue
        videos_dir = base / "videos"
        if videos_dir.exists():
            roots.append(videos_dir)
        roots.append(base)
    return roots


def make_video_id(dataset: str, path: Path, raw_root: Path):
    rel = path.relative_to(raw_root).as_posix()
    safe = rel.replace("/", "__").replace(":", "")
    return f"{dataset}__{safe}"


def read_test_list(raw_root: Path):
    """
    List_of_testing_videos.txt commonly contains lines like:
      YouTube-real/00238.mp4
      Celeb-real/00001.mp4
      Celeb-synthesis/00001.mp4

    Sometimes datasets store files under <split>/videos/*.mp4.
    We normalize both forms for matching.
    """
    test_file = raw_root / "List_of_testing_videos.txt"
    if not test_file.exists():
        return set(), set()

    lines = []
    for line in test_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        # remove leading ./ or /
        while line.startswith("./"):
            line = line[2:]
        while line.startswith("/"):
            line = line[1:]
        lines.append(line)

    # Two matching keys:
    # 1) exact relative path as written (e.g., "YouTube-real/00238.mp4")
    # 2) with "videos/" inserted (e.g., "YouTube-real/videos/00238.mp4")
    test_rel = set()
    test_rel_with_videos = set()

    for rel in lines:
        test_rel.add(rel.replace("\\", "/"))
        parts = rel.replace("\\", "/").split("/")
        if len(parts) >= 2:
            test_rel_with_videos.add("/".join([parts[0], "videos"] + parts[1:]))

    return test_rel, test_rel_with_videos


def infer_label_and_source(video_path: Path, raw_root: Path):
    """
    Real:
      Celeb-real, YouTube-real
    Fake:
      Celeb-synthesis
    """
    rel_parts = video_path.relative_to(raw_root).parts
    top = rel_parts[0] if len(rel_parts) > 0 else ""
    if top == "Celeb-synthesis":
        return 1, top
    if top in {"Celeb-real", "YouTube-real"}:
        return 0, top
    # fallback: best-effort by name
    return (1 if "synth" in top.lower() else 0), top


def split_train_val(items, seed=42, val_ratio=0.1):
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_val = int(val_ratio * n)
    val = items[:n_val]
    train = items[n_val:]
    return train, val


def main():
    dataset = "CelebDFv2"
    raw_root = Path("data/raw/Celeb-DF-v2")

    # Gather all videos
    roots = dataset_video_roots(raw_root)
    videos = []
    seen = set()
    for r in roots:
        for p in list_videos(r):
            if p in seen:
                continue
            seen.add(p)
            # only accept if it's truly under raw_root
            try:
                _ = p.relative_to(raw_root)
            except ValueError:
                continue
            videos.append(p)

    # Read official test list
    test_rel, test_rel_with_videos = read_test_list(raw_root)

    rows = []
    for vp in tqdm(videos, desc="Scanning Celeb-DF v2"):
        label, source = infer_label_and_source(vp, raw_root)
        rel = vp.relative_to(raw_root).as_posix()

        # match official test list both ways:
        # - stored rel might be ".../videos/....mp4"
        # - list might omit "videos/"
        rel_no_videos = rel.replace("/videos/", "/")

        is_test = (rel in test_rel) or (rel in test_rel_with_videos) or (rel_no_videos in test_rel)

        rows.append({
            "dataset": dataset,
            "video_id": make_video_id(dataset, vp, raw_root),
            "video_path": str(vp),
            "label": label,
            "source": source,
            "is_protocol_test": int(is_test),
        })

    df = pd.DataFrame(rows)

    out_manifest = Path("data/metadata/celebdfv2_manifest.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print("\nSaved manifest:", out_manifest)
    print("Total rows:", len(df))
    print("Label distribution:\n", df["label"].value_counts(dropna=False))
    print("Source distribution:\n", df["source"].value_counts(dropna=False))
    print("Protocol TEST count:", int(df["is_protocol_test"].sum()))

    # Build splits:
    # - TEST: official protocol list (if present)
    # - TRAIN/VAL: split remaining
    test_df = df[df["is_protocol_test"] == 1].copy()
    rest_df = df[df["is_protocol_test"] == 0].copy()

    rest_ids = rest_df["video_id"].tolist()
    train_ids, val_ids = split_train_val(rest_ids, seed=42, val_ratio=0.1)

    Path("data/splits").mkdir(parents=True, exist_ok=True)

    def save_split(split_name, split_df):
        out_path = Path("data/splits") / f"celebdfv2_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"Saved split: {out_path}  rows={len(split_df)}")

    save_split("test", test_df)
    save_split("train", rest_df[rest_df["video_id"].isin(train_ids)].copy())
    save_split("val", rest_df[rest_df["video_id"].isin(val_ids)].copy())


if __name__ == "__main__":
    main()
