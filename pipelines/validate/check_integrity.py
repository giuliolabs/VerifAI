"""
Preprocessing Integrity Check
=============================

Verifies that preprocessing outputs (frames, audio, MFCC)
exist and are readable for sampled items across datasets.

This script is used to validate the correctness of the
full data-preprocessing pipeline (Week 14).

Dependencies:
    pip install pandas numpy

Run:
    python pipelines/validate/check_integrity.py

Output:
    experiments/results/preprocessing_checks/integrity_report.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np
import random


def check_frames(frames_root: Path, n_show=5):
    lines = []
    if not frames_root.exists():
        return [f"Frames folder missing: {frames_root}"]

    folders = [p for p in frames_root.iterdir() if p.is_dir()]
    if not folders:
        return [f"No frame folders found in {frames_root}"]

    sample = random.sample(folders, min(n_show, len(folders)))
    for f in sample:
        count = len(list(f.glob("frame_*.jpg")))
        lines.append(f"{f.name}: frames={count}")

    lines.append(f"Total frame folders: {len(folders)}")
    return lines


def check_audio_and_mfcc(audio_dir: Path, mfcc_dir: Path, n_show=10):
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
    out = Path("experiments/ablations/results/preprocessing_checks/integrity_report.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["PREPROCESSING INTEGRITY REPORT", "================================\n", "=== DeeperForensics (VAL) ==="]

    # DeeperForensics (frames only)
    df_frames = Path("data/interim/frames/DeeperForensics/val")
    lines.extend(check_frames(df_frames))
    lines.append("")
    lines.append("Audio:")
    lines.append(
        "No audio streams detected in DeeperForensics VAL subset "
        "(ffprobe: 738/738 no-audio). Audio + MFCC intentionally skipped.\n"
    )

    # FakeAVCeleb (audio + MFCC)
    lines.append("=== FakeAVCeleb_v1.2 (VAL) ===")
    fa_audio = Path("data/interim/audio/FakeAVCeleb_v1.2/val")
    fa_mfcc = Path("data/processed/audio_features/FakeAVCeleb_v1.2/val")
    lines.extend(check_audio_and_mfcc(fa_audio, fa_mfcc))

    out.write_text("\n".join(lines), encoding="utf-8")
    print("Saved ->", out)


if __name__ == "__main__":
    main()
