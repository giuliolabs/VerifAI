"""
Generic Audio Feature Extraction (MFCC)
======================================

Generates MFCC features from WAV files.
Dataset-agnostic: works for all datasets that share
a common audio directory structure.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
pip install librosa soundfile numpy tqdm

------------------------------------------------
INPUT
------------------------------------------------
Directory containing WAV files:
    data/interim/audio/<Dataset>/<split>/*.wav

------------------------------------------------
OUTPUT
------------------------------------------------
MFCC feature files:
    data/processed/audio_features/<Dataset>/<split>/<safe_id>.npy

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- MFCCs are computed at 16kHz to match audio extraction.
- 40 coefficients are used (common in AV deepfake literature).
- Output is stored as NumPy arrays for efficient loading.
- Script is resume-safe.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import argparse
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm


def compute_mfcc(wav_path: Path, sr=16000, n_mfcc=40):
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--n_mfcc", type=int, default=40)
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        print("No WAV files found in:", wav_dir)
        return

    saved = skipped = 0

    for wav_path in tqdm(wav_files, desc=f"MFCC -> {out_dir}"):
        out_path = out_dir / f"{wav_path.stem}.npy"

        if out_path.exists():
            skipped += 1
            continue

        mfcc = compute_mfcc(wav_path, n_mfcc=args.n_mfcc)
        np.save(out_path, mfcc)
        saved += 1

    print("\nDone.")
    print("Saved:", saved)
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()
