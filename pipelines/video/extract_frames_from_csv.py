"""
Extract N Frames per Video from a Split CSV (Dataset-Agnostic)
==============================================================

This script extracts a fixed number of frames from each video listed in
a dataset split CSV file. It is designed to be **dataset-agnostic** and
**Windows-safe**, meaning it works across different datasets used in
VerifAI without requiring dataset-specific code.

The script reads a CSV manifest containing video paths and IDs, samples
frames evenly across each video, and saves them as resized images.

------------------------------------------------
DEPENDENCIES
------------------------------------------------

Install required packages:

    pip install opencv-python pandas tqdm

------------------------------------------------
INPUT
------------------------------------------------

Split CSV file containing at least the columns:

    video_path, video_id

Example:

    data/splits/fakeavceleb_val.csv
    data/splits/deeperforensics_val.csv

------------------------------------------------
OUTPUT
------------------------------------------------

Frames are saved to:

    <out_dir>/<SAFE_VIDEO_ID>/frame_000.jpg
    <out_dir>/<SAFE_VIDEO_ID>/frame_001.jpg
    ...

Example:

    data/interim/frames/FakeAVCeleb_v1.2/val/3f42ac91b1e2f3a4/frame_000.jpg

A mapping file is also created:

    _id_map.csv

This file records:

    safe_id,video_id,video_path

This allows the hashed safe IDs to be traced back to the original video.

------------------------------------------------
WHY SAFE_ID IS USED
------------------------------------------------

Windows systems (especially with OneDrive) can hit MAX_PATH errors when
filenames are long. To prevent this, each video is assigned a short,
stable identifier:

    safe_id = SHA1(video_id)[:16]

This ensures:

- shorter folder paths
- deterministic naming
- compatibility across systems

------------------------------------------------
RUN EXAMPLES
------------------------------------------------

Example with DeeperForensics:

python pipelines/video/extract_frames_from_csv.py \
    --split_csv data/splits/deeperforensics_val.csv \
    --out_dir data/interim/frames/DeeperForensics/val \
    --frames_per_video 5


Example with FakeAVCeleb:

python pipelines/video/extract_frames_from_csv.py \
    --split_csv data/splits/fakeavceleb_val.csv \
    --out_dir data/interim/frames/FakeAVCeleb_v1.2/val \
    --frames_per_video 5

------------------------------------------------
NOTES
------------------------------------------------

- The script is **resume-safe**: already processed videos are skipped.
- Frame sampling is **uniform across the video duration**.
- If a video cannot be opened or contains no frames, it is counted as
  a failed sample.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
"""

import argparse
from pathlib import Path

# OpenCV is used for reading video frames
import cv2

# pandas reads the split CSV manifest
import pandas as pd

# tqdm provides a progress bar
from tqdm import tqdm


def sample_frame_indices(frame_count: int, n: int):
    """
    Compute evenly spaced frame indices across a video.

    Args:
        frame_count: total number of frames in the video
        n: number of frames to extract

    Returns:
        List of frame indices to sample.
    """

    if frame_count <= 0:
        return []

    # If the video contains fewer frames than requested,
    # return all frame indices.
    if frame_count <= n:
        return list(range(frame_count))

    # Otherwise compute uniform positions across the video
    return [int(i * (frame_count - 1) / (n - 1)) for i in range(n)]


def make_safe_id(video_id: str) -> str:
    """
    Generate a short, Windows-safe ID from the video ID.

    This avoids MAX_PATH issues and ensures reproducible
    folder naming.

    safe_id = SHA1(video_id)[:16]
    """

    import hashlib

    return hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:16]


def extract_frames(video_path: str, out_folder: Path, n_frames: int, resize=(224, 224)):
    """
    Extract N frames from a video and save them to disk.

    Args:
        video_path: path to the input video
        out_folder: directory where frames will be saved
        n_frames: number of frames to extract
        resize: output resolution

    Returns:
        Tuple (success_flag, message)
    """

    out_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False, "cannot_open"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = sample_frame_indices(frame_count, n_frames)

    saved = 0
    idx_set = set(indices)
    current = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current in idx_set:

            frame = cv2.resize(frame, resize)

            out_path = out_folder / f"frame_{saved:03d}.jpg"

            cv2.imwrite(str(out_path), frame)

            saved += 1

        current += 1

    cap.release()

    if saved == 0:
        return False, "no_frames"

    return True, f"saved_{saved}"


def main():
    """
    Main program logic.

    Steps:
        1. Read split CSV
        2. Generate safe IDs
        3. Extract frames
        4. Save mapping file
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--split_csv", required=True, type=str)

    parser.add_argument("--out_dir", required=True, type=str)

    parser.add_argument("--frames_per_video", type=int, default=5)

    args = parser.parse_args()

    # Load dataset split CSV
    df = pd.read_csv(args.split_csv)

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Mapping file linking safe_id to original video
    map_path = out_dir / "_id_map.csv"

    if not map_path.exists():
        map_path.write_text("safe_id,video_id,video_path\n", encoding="utf-8")

    extracted = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting frames -> {out_dir}"):

        video_path = str(row["video_path"])

        video_id = str(row["video_id"])

        safe_id = make_safe_id(video_id)

        # Record mapping between safe_id and original video
        with open(map_path, "a", encoding="utf-8") as f:
            f.write(f"{safe_id},{video_id},{video_path}\n")

        vid_out = out_dir / safe_id

        # Resume-safe: skip if frames already extracted
        if vid_out.exists() and len(list(vid_out.glob("frame_*.jpg"))) >= args.frames_per_video:
            skipped += 1
            continue

        if not Path(video_path).exists():
            failed += 1
            continue

        ok, _ = extract_frames(video_path, vid_out, args.frames_per_video)

        if ok:
            extracted += 1
        else:
            failed += 1

    print("\nDone.")
    print("Extracted:", extracted)
    print("Skipped:", skipped)
    print("Failed:", failed)


if __name__ == "__main__":
    main()