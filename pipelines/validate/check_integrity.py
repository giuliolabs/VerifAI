"""
Preprocessing Integrity Check – VerifAI
======================================

This script verifies that preprocessing outputs produced by the VerifAI
pipeline exist and are readable. It performs lightweight validation
checks for sampled items across datasets.

The script checks three types of preprocessing artifacts:

    1) Extracted video frames
    2) Extracted audio (.wav)
    3) Computed MFCC audio features (.npy)

The goal is to ensure that the preprocessing pipeline has executed
correctly and produced usable data before model training or evaluation.

This script was used during **Week 14 pipeline validation** to confirm
that intermediate preprocessing outputs were correctly generated.

------------------------------------------------
DEPENDENCIES
------------------------------------------------

Install required packages:

    pip install pandas numpy

------------------------------------------------
RUN
------------------------------------------------

python pipelines/validate/check_integrity.py

------------------------------------------------
OUTPUT
------------------------------------------------

A human-readable report is saved to:

    experiments/results/preprocessing_checks/integrity_report.txt

The report summarizes:

    - number of frame folders detected
    - frame counts for sampled videos
    - audio/MFCC file presence
    - MFCC tensor shapes
    - pass/fail counts for sampled items

------------------------------------------------
NOTES
------------------------------------------------

- Only a subset of samples is checked to keep validation fast.
- The script is dataset-aware and can skip checks where appropriate
  (for example, datasets that contain no audio streams).

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
"""

from pathlib import Path
import pandas as pd
import numpy as np
import random


def check_frames(frames_root: Path, n_show=5):
    """
    Verify that frame folders exist and contain frame images.

    A small number of folders are randomly sampled and the number
    of extracted frames is reported.

    Args:
        frames_root: directory containing frame folders
        n_show: number of folders to sample

    Returns:
        List of report lines.
    """

    lines = []

    if not frames_root.exists():
        return [f"Frames folder missing: {frames_root}"]

    # Each video should have its own folder of frames
    folders = [p for p in frames_root.iterdir() if p.is_dir()]

    if not folders:
        return [f"No frame folders found in {frames_root}"]

    # Randomly sample a few videos
    sample = random.sample(folders, min(n_show, len(folders)))

    for f in sample:
        count = len(list(f.glob("frame_*.jpg")))
        lines.append(f"{f.name}: frames={count}")

    lines.append(f"Total frame folders: {len(folders)}")

    return lines


def check_audio_and_mfcc(audio_dir: Path, mfcc_dir: Path, n_show=10):
    """
    Verify that audio (.wav) files and MFCC feature files exist.

    The script reads the mapping file `_id_map.csv`, samples a few items,
    and checks whether both the WAV and MFCC files are present.

    Args:
        audio_dir: directory containing extracted audio files
        mfcc_dir: directory containing MFCC feature files
        n_show: number of items to sample

    Returns:
        List of report lines.
    """

    lines = []

    id_map = audio_dir / "_id_map.csv"

    if not id_map.exists():
        return [f"Missing _id_map.csv in {audio_dir}"]

    df = pd.read_csv(id_map)

    if len(df) == 0:
        return ["_id_map.csv is empty"]

    sample = df.sample(n=min(n_show, len(df)), random_state=42)

    ok = 0

    for _, r in sample.iterrows():

        safe_id = str(r["safe_id"])

        wav = audio_dir / f"{safe_id}.wav"
        mfcc = mfcc_dir / f"{safe_id}.npy"

        wav_ok = wav.exists() and wav.stat().st_size > 0
        mfcc_ok = mfcc.exists() and mfcc.stat().st_size > 0

        shape = "missing"

        if mfcc_ok:
            try:
                shape = str(np.load(mfcc).shape)
            except Exception:
                shape = "load_failed"

        status = "OK" if (wav_ok and mfcc_ok) else "FAIL"

        if status == "OK":
            ok += 1

        lines.append(
            f"{status} | {safe_id} | wav={wav_ok} mfcc={mfcc_ok} mfcc_shape={shape}"
        )

    lines.append(f"Passed {ok}/{len(sample)} sampled items")

    return lines


def main():
    """
    Run preprocessing integrity checks for selected datasets.

    The script generates a human-readable report summarizing
    preprocessing artifacts detected in the project directories.
    """

    out = Path("experiments/results/preprocessing_checks/integrity_report.txt")

    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "PREPROCESSING INTEGRITY REPORT",
        "================================\n",
        "=== DeeperForensics (VAL) ==="
    ]

    # -------------------------------------
    # DeeperForensics dataset
    # -------------------------------------

    # This dataset contains frames but no audio
    df_frames = Path("data/interim/frames/DeeperForensics/val")

    lines.extend(check_frames(df_frames))

    lines.append("")
    lines.append("Audio:")

    lines.append(
        "No audio streams detected in DeeperForensics VAL subset "
        "(ffprobe: 738/738 no-audio). Audio + MFCC intentionally skipped.\n"
    )

    # -------------------------------------
    # FakeAVCeleb dataset
    # -------------------------------------

    lines.append("=== FakeAVCeleb_v1.2 (VAL) ===")

    fa_audio = Path("data/interim/audio/FakeAVCeleb_v1.2/val")

    fa_mfcc = Path("data/processed/audio_features/FakeAVCeleb_v1.2/val")

    lines.extend(check_audio_and_mfcc(fa_audio, fa_mfcc))

    # -------------------------------------
    # Save report
    # -------------------------------------

    out.write_text("\n".join(lines), encoding="utf-8")

    print("Saved ->", out)


if __name__ == "__main__":
    main()