"""
Generic Video Feature Extraction (Frame-Stats)
=============================================

Creates a small, deterministic feature vector per video by reading the
already-extracted frames for each sample.

INPUT
-----
Frames directory:
    data/interim/frames/<Dataset>/<split>/<safe_id>/frame_*.jpg

OUTPUT
------
Feature files:
    data/processed/video_features/<Dataset>/<split>/<safe_id>.npy

Notes
-----
- Dataset-agnostic.
- Resume-safe.
- Windows-safe (uses safe_id folders already produced by frame extraction).

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm


def compute_features_from_frames(sample_dir: Path) -> np.ndarray | None:
    frame_paths = sorted(sample_dir.glob("frame_*.jpg"))
    if not frame_paths:
        return None

    rgb_means = []
    rgb_stds = []
    gray_means = []
    gray_stds = []
    sharpness = []

    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_means.append(img_rgb.mean(axis=(0, 1)))
        rgb_stds.append(img_rgb.std(axis=(0, 1)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_means.append(float(gray.mean()))
        gray_stds.append(float(gray.std()))

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness.append(float(lap.var()))

    if not rgb_means:
        return None

    rgb_means = np.stack(rgb_means, axis=0).mean(axis=0)   # (3,)
    rgb_stds = np.stack(rgb_stds, axis=0).mean(axis=0)     # (3,)
    gray_mean = float(np.mean(gray_means))
    gray_std = float(np.mean(gray_stds))
    sharp = float(np.mean(sharpness))

    feat = np.array(
        [rgb_means[0], rgb_means[1], rgb_means[2],
         rgb_stds[0], rgb_stds[1], rgb_stds[2],
         gray_mean, gray_std, sharp],
        dtype=np.float32
    )
    return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True, type=str,
                        help="e.g. data/interim/frames/FakeAVCeleb_v1.2/train")
    parser.add_argument("--out_dir", required=True, type=str,
                        help="e.g. data/processed/video_features/FakeAVCeleb_v1.2/train")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir()])
    if not sample_dirs:
        print("No sample folders found in:", frames_dir)
        return

    saved = skipped = failed = 0

    for sample_dir in tqdm(sample_dirs, desc=f"Video feats -> {out_dir}"):
        safe_id = sample_dir.name
        out_path = out_dir / f"{safe_id}.npy"

        if out_path.exists():
            skipped += 1
            continue

        feat = compute_features_from_frames(sample_dir)
        if feat is None:
            failed += 1
            continue

        np.save(out_path, feat)
        saved += 1

    print("\nDone.")
    print("Saved:", saved)
    print("Skipped:", skipped)
    print("Failed:", failed)


if __name__ == "__main__":
    main()
