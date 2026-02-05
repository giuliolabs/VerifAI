"""
Frame Extraction Script – FaceForensics++ (C23)
==============================================

This script extracts a fixed number of frames per video (uniformly sampled)
from FaceForensics++ videos, using pre-defined train/val/test splits.

It is designed for academic reproducibility and lightweight submission,
avoiding full video frame dumps.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
Required Python packages (install via pip):

    pip install opencv-python pandas tqdm

Tested with:
- Python 3.10+
- OpenCV 4.x
- pandas 2.x

------------------------------------------------
INPUT
------------------------------------------------
CSV files (created beforehand by pipelines/splits/make_splits.py):

    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv

Each CSV must contain columns:
    video_path,label,video_id

------------------------------------------------
OUTPUT
------------------------------------------------
Extracted frames are saved to:

    data/interim/frames/FaceForensics++_C23/{split}/{real|fake}/{video_id}/

Each video folder contains exactly N_FRAMES JPEG images:
    frame_000001.jpg ... frame_000005.jpg

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- Frames are sampled uniformly across the video duration.
- Extraction is resume-safe: videos already processed are skipped.
- Only a small representative subset of frames is extracted
  to reduce storage requirements, in line with common deepfake
  detection baselines.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm


# ==============================
# CONFIGURATION
# ==============================

N_FRAMES = 5  # Number of frames per video (uniformly sampled)


# ==============================
# UTILITY FUNCTIONS
# ==============================

def ensure_dir(path: Path):
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def existing_frame_count(out_dir: Path) -> int:
    """Count already-extracted frames (for resume safety)."""
    return len(list(out_dir.glob("frame_*.jpg")))


def get_total_frames(cap: cv2.VideoCapture) -> int:
    """Return total frame count if available, otherwise -1."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total if total > 0 else -1


def uniform_indices(total_frames: int, n_frames: int):
    """
    Compute uniformly spaced frame indices.
    Example: total=100, n=5 -> [0, 24, 49, 74, 99]
    """
    if total_frames <= n_frames:
        return list(range(total_frames))

    indices = []
    for i in range(n_frames):
        idx = round(i * (total_frames - 1) / (n_frames - 1))
        indices.append(int(idx))

    return sorted(set(indices))


def extract_frames(video_path: Path, out_dir: Path, n_frames: int) -> int:
    """Extract n_frames uniformly sampled frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total_frames = get_total_frames(cap)
    saved = 0

    if total_frames > 0:
        indices = uniform_indices(total_frames, n_frames)
        for i, frame_idx in enumerate(indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            out_path = out_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
    else:
        # Fallback for videos with unreliable metadata
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

        if frames:
            indices = uniform_indices(len(frames), n_frames)
            for i, idx in enumerate(indices, start=1):
                out_path = out_dir / f"frame_{i:06d}.jpg"
                cv2.imwrite(str(out_path), frames[idx])
                saved += 1

    cap.release()
    return saved


# ==============================
# MAIN PROCESSING LOGIC
# ==============================

def process_split(split_name: str):
    split_csv = Path("data/splits") / f"{split_name}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(split_csv)

    df = pd.read_csv(split_csv)

    required_cols = {"video_path", "label", "video_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{split_csv} must contain columns: {required_cols}")

    output_root = Path("data/interim/frames/FaceForensics++_C23") / split_name
    ensure_dir(output_root)

    extracted = skipped = failed = 0

    print(f"\n=== Extracting {N_FRAMES} frames/video | split: {split_name} ===")

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        video_path = Path(row.video_path)
        label = int(row.label)
        video_id = str(row.video_id)

        class_name = "real" if label == 0 else "fake"
        out_dir = output_root / class_name / video_id
        ensure_dir(out_dir)

        # Resume-safe check
        if existing_frame_count(out_dir) >= N_FRAMES:
            skipped += 1
            continue

        if not video_path.exists():
            failed += 1
            continue

        saved = extract_frames(video_path, out_dir, N_FRAMES)
        if saved == 0:
            failed += 1
        else:
            extracted += 1

    print(f"Completed split: {split_name}")
    print(f"  Videos extracted: {extracted}")
    print(f"  Videos skipped:   {skipped}")
    print(f"  Videos failed:    {failed}")


def main():
    for split in ["train", "val", "test"]:
        process_split(split)


if __name__ == "__main__":
    main()
