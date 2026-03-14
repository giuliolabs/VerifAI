"""
Generic Audio Feature Extraction (MFCC)
======================================

This script generates MFCC (Mel-Frequency Cepstral Coefficient) features
from WAV audio files. It is designed to be dataset-agnostic, meaning it
can be reused for any dataset that follows the same audio folder layout.

MFCCs are a compact representation of audio content and are widely used
in speech processing and audio deepfake detection because they capture
important frequency characteristics in a way that is suitable for machine
learning models.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
Install required packages with:

    pip install librosa soundfile numpy tqdm

------------------------------------------------
INPUT
------------------------------------------------
Directory containing WAV files:

    data/interim/audio/<Dataset>/<split>/*.wav

------------------------------------------------
OUTPUT
------------------------------------------------
MFCC feature files saved as NumPy arrays:

    data/processed/audio_features/<Dataset>/<split>/<safe_id>.npy

Each output file contains a 2D MFCC matrix with shape:

    (n_mfcc, time_steps)

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- MFCCs are computed at 16kHz to match the audio extraction stage.
- 40 coefficients are used, which is common in audio-visual deepfake literature.
- Output is stored as NumPy arrays for efficient training-time loading.
- The script is resume-safe, so existing feature files are skipped.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# argparse is used to read command-line arguments
import argparse

# Path is used for clean and platform-independent file handling
from pathlib import Path

# NumPy is used to save MFCC feature arrays as .npy files
import numpy as np

# librosa is used to load audio and compute MFCC features
import librosa

# tqdm provides a progress bar during feature extraction
from tqdm import tqdm


def compute_mfcc(wav_path: Path, sr=16000, n_mfcc=40):
    """
    Load one WAV file and compute its MFCC features.

    Parameters:
        wav_path : Path
            Path to the input WAV file.

        sr : int
            Target sample rate. Default is 16000 Hz to match the
            earlier audio extraction pipeline.

        n_mfcc : int
            Number of MFCC coefficients to compute.

    Returns:
        np.ndarray
            MFCC feature matrix with dtype float32.
    """

    # Load audio as mono and resample if necessary
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

    # Compute MFCC features from the waveform
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Convert to float32 to keep storage smaller and consistent
    return mfcc.astype(np.float32)


def main():
    """
    Main execution function.

    This function:
    1. Reads command-line arguments
    2. Finds all WAV files in the input directory
    3. Computes MFCC features for each file
    4. Saves each result as a .npy file
    5. Skips files that already exist (resume-safe)
    """

    parser = argparse.ArgumentParser()

    # Input folder containing WAV files
    parser.add_argument("--wav_dir", required=True, type=str)

    # Output folder where MFCC .npy files will be stored
    parser.add_argument("--out_dir", required=True, type=str)

    # Number of MFCC coefficients to generate
    parser.add_argument("--n_mfcc", type=int, default=40)

    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)

    # Create output directory if it does not already exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all WAV files in sorted order for reproducibility
    wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        print("No WAV files found in:", wav_dir)
        return

    saved = 0
    skipped = 0

    for wav_path in tqdm(wav_files, desc=f"MFCC -> {out_dir}"):
        # Output file uses the same stem as the WAV file
        out_path = out_dir / f"{wav_path.stem}.npy"

        # Resume-safe behavior:
        # if the output already exists, skip recomputing it
        if out_path.exists():
            skipped += 1
            continue

        # Compute MFCC features
        mfcc = compute_mfcc(wav_path, n_mfcc=args.n_mfcc)

        # Save as NumPy array
        np.save(out_path, mfcc)
        saved += 1

    print("\nDone.")
    print("Saved:", saved)
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()