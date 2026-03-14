"""
Frame Extraction Script – FaceForensics++ (C23)
==============================================

This script extracts a fixed number of frames per video from the
FaceForensics++ C23 dataset. The frames are sampled uniformly across
the full video so that each sample still represents the video content
without saving every single frame.

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
CSV files created beforehand by pipelines/splits/make_splits.py:

    data/splits/faceforensics++_c23_train.csv
    data/splits/faceforensics++_c23_val.csv
    data/splits/faceforensics++_c23_test.csv

Each CSV must contain columns:
    video_path,label,video_id

------------------------------------------------
OUTPUT
------------------------------------------------
Extracted frames are saved to:

    data/interim/frames/FaceForensics++_C23/{split}/{real|fake}/{video_id}/

Each video folder contains up to N_FRAMES JPEG images:
    frame_000001.jpg ... frame_000005.jpg

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- Frames are sampled uniformly across the video duration.
- Extraction is resume-safe: videos already processed are skipped.
- Only a small representative subset of frames is extracted
  to reduce storage requirements, following common deepfake
  detection baselines.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Path is used for clean and platform-independent file handling
from pathlib import Path

# OpenCV is used to open videos and save frames
import cv2

# pandas is used to read the split CSV files
import pandas as pd

# tqdm adds a progress bar during extraction
from tqdm import tqdm


# ==============================
# CONFIGURATION
# ==============================

# Number of frames to extract from each video
N_FRAMES = 5


# ==============================
# UTILITY FUNCTIONS
# ==============================

def ensure_dir(path: Path):
    """
    Create a directory if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def existing_frame_count(out_dir: Path) -> int:
    """
    Count how many frames are already present in an output folder.

    This helps make the script resume-safe, so already processed
    videos do not need to be extracted again.
    """
    return len(list(out_dir.glob("frame_*.jpg")))


def get_total_frames(cap: cv2.VideoCapture) -> int:
    """
    Return the total number of frames in a video if available.

    If OpenCV cannot read a valid frame count, return -1 so
    the script can switch to a fallback method.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total if total > 0 else -1


def uniform_indices(total_frames: int, n_frames: int):
    """
    Compute uniformly spaced frame indices across a video.

    Example:
        total=100, n=5 -> [0, 25, 50, 74, 99]

    If the video has fewer frames than requested, all frame
    indices are returned instead.
    """
    if total_frames <= n_frames:
        return list(range(total_frames))

    indices = []

    for i in range(n_frames):
        idx = round(i * (total_frames - 1) / (n_frames - 1))
        indices.append(int(idx))

    # sorted(set(...)) avoids duplicate indices in edge cases
    return sorted(set(indices))


def extract_frames(video_path: Path, out_dir: Path, n_frames: int) -> int:
    """
    Extract a fixed number of uniformly sampled frames from a video.

    Returns:
        Number of frames successfully saved.
    """
    cap = cv2.VideoCapture(str(video_path))

    # If the video cannot be opened, extraction fails immediately
    if not cap.isOpened():
        return 0

    total_frames = get_total_frames(cap)
    saved = 0

    # Preferred case: frame count metadata is available
    if total_frames > 0:
        indices = uniform_indices(total_frames, n_frames)

        for i, frame_idx in enumerate(indices, start=1):
            # Jump directly to the chosen frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()

            if not ok:
                continue

            out_path = out_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

    else:
        # Fallback case:
        # Some videos may have unreliable metadata, so we read
        # all frames first and then sample from that list.
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
    """
    Process one dataset split (train, val, or test).

    For each video listed in the split CSV:
    - check that the file exists
    - create its output folder
    - skip it if frames already exist
    - extract uniformly sampled frames
    """
    split_csv = Path("data/splits") / f"{split_name}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(split_csv)

    df = pd.read_csv(split_csv)

    # Make sure the split file contains the columns the script needs
    required_cols = {"video_path", "label", "video_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{split_csv} must contain columns: {required_cols}")

    output_root = Path("data/interim/frames/FaceForensics++_C23") / split_name
    ensure_dir(output_root)

    extracted = 0
    skipped = 0
    failed = 0

    print(f"\n=== Extracting {N_FRAMES} frames/video | split: {split_name} ===")

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        video_path = Path(row.video_path)
        label = int(row.label)
        video_id = str(row.video_id)

        # Convert numeric label into folder name for readability
        class_name = "real" if label == 0 else "fake"
        out_dir = output_root / class_name / video_id
        ensure_dir(out_dir)

        # Resume-safe check:
        # if enough frames already exist, skip this video
        if existing_frame_count(out_dir) >= N_FRAMES:
            skipped += 1
            continue

        # If the source video does not exist, count it as failed
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
    """
    Run frame extraction for all standard dataset splits.
    """
    for split in ["train", "val", "test"]:
        process_split(split)


if __name__ == "__main__":
    main()