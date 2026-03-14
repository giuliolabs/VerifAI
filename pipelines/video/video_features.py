"""
Generic Video Feature Extraction (Frame-Stats)
=============================================

This script creates a small, deterministic feature vector for each video
by analysing the frames that were already extracted during preprocessing.

Instead of running a deep model, it computes simple statistical features
from the saved frames. This makes the script lightweight, reproducible,
and useful for baseline experiments or metadata-style feature analysis.

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
- Windows-safe.

The feature vector includes:
- mean RGB values
- standard deviation of RGB values
- grayscale mean
- grayscale standard deviation
- average sharpness estimate

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

import argparse
from pathlib import Path

# NumPy is used for feature vectors and saving .npy files
import numpy as np

# OpenCV is used to read images and compute grayscale / sharpness statistics
import cv2

# tqdm provides a progress bar during feature extraction
from tqdm import tqdm


def compute_features_from_frames(sample_dir: Path) -> np.ndarray | None:
    """
    Compute a fixed statistical feature vector from all frames in one sample folder.

    Each sample folder is expected to contain files like:
        frame_000.jpg, frame_001.jpg, ...

    Returns:
        np.ndarray of shape (9,) if successful
        None if no valid frames could be processed
    """
    frame_paths = sorted(sample_dir.glob("frame_*.jpg"))
    if not frame_paths:
        return None

    # Per-frame statistics are collected first,
    # then averaged across the full video sample.
    rgb_means = []
    rgb_stds = []
    gray_means = []
    gray_stds = []
    sharpness = []

    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue

        # Convert from OpenCV default BGR format to RGB
        # so color statistics are easier to interpret.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mean and standard deviation for each RGB channel
        rgb_means.append(img_rgb.mean(axis=(0, 1)))
        rgb_stds.append(img_rgb.std(axis=(0, 1)))

        # Grayscale statistics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_means.append(float(gray.mean()))
        gray_stds.append(float(gray.std()))

        # Simple sharpness estimate using Laplacian variance.
        # Higher variance usually means more edges / detail.
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness.append(float(lap.var()))

    # If all frames failed to load, return None
    if not rgb_means:
        return None

    # Average per-frame statistics to get one stable feature vector per video
    rgb_means = np.stack(rgb_means, axis=0).mean(axis=0)   # shape (3,)
    rgb_stds = np.stack(rgb_stds, axis=0).mean(axis=0)     # shape (3,)
    gray_mean = float(np.mean(gray_means))
    gray_std = float(np.mean(gray_stds))
    sharp = float(np.mean(sharpness))

    # Final feature vector layout:
    # [R_mean, G_mean, B_mean, R_std, G_std, B_std, gray_mean, gray_std, sharpness]
    feat = np.array(
        [
            rgb_means[0], rgb_means[1], rgb_means[2],
            rgb_stds[0], rgb_stds[1], rgb_stds[2],
            gray_mean, gray_std, sharp
        ],
        dtype=np.float32
    )
    return feat


def main():
    """
    Main execution function.

    This function:
    - reads all sample folders in the input frames directory
    - computes one feature vector per sample
    - saves each vector as <safe_id>.npy
    - skips files that already exist so the script is resume-safe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames_dir",
        required=True,
        type=str,
        help="e.g. data/interim/frames/FakeAVCeleb_v1.2/train"
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="e.g. data/processed/video_features/FakeAVCeleb_v1.2/train"
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each sample is stored inside its own folder named by safe_id
    sample_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir()])
    if not sample_dirs:
        print("No sample folders found in:", frames_dir)
        return

    saved = 0
    skipped = 0
    failed = 0

    for sample_dir in tqdm(sample_dirs, desc=f"Video feats -> {out_dir}"):
        safe_id = sample_dir.name
        out_path = out_dir / f"{safe_id}.npy"

        # Skip samples already processed
        if out_path.exists():
            skipped += 1
            continue

        feat = compute_features_from_frames(sample_dir)

        # If no valid feature vector could be built, count as failed
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