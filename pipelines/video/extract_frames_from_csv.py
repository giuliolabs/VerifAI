"""
Extract N frames per video from a split CSV (dataset-agnostic, Windows-safe).

Dependencies:
    pip install opencv-python pandas tqdm

Run examples:
    python pipelines/video/extract_frames_from_csv.py --split_csv data/splits/deeperforensics_val.csv --out_dir
    data/interim/frames/DeeperForensics/val --frames_per_video 5

    python pipelines/video/extract_frames_from_csv.py --split_csv data/splits/fakeavceleb_val.csv --out_dir
    data/interim/frames/FakeAVCeleb_v1.2/val --frames_per_video 5

Output:
    <out_dir>/<SAFE_VIDEO_ID>/frame_000.jpg ... frame_004.jpg
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def sample_frame_indices(frame_count: int, n: int):
    if frame_count <= 0:
        return []
    if frame_count <= n:
        return list(range(frame_count))
    return [int(i * (frame_count - 1) / (n - 1)) for i in range(n)]


def make_safe_id(video_id: str) -> str:
    """
    Ultra-short Windows-safe ID to avoid MAX_PATH issues on OneDrive.
    Uses a stable hash so it's reproducible.
    """
    import hashlib
    return hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:16]



def extract_frames(video_path: str, out_folder: Path, n_frames: int, resize=(224, 224)):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--frames_per_video", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / "_id_map.csv"
    if not map_path.exists():
        map_path.write_text("safe_id,video_id,video_path\n", encoding="utf-8")

    extracted, skipped, failed = 0, 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting frames -> {out_dir}"):
        video_path = str(row["video_path"])
        video_id = str(row["video_id"])

        safe_id = make_safe_id(video_id)
        with open(map_path, "a", encoding="utf-8") as f:
            f.write(f"{safe_id},{video_id},{video_path}\n")
        vid_out = out_dir / safe_id

        # resume safe
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
