"""
Generic Audio Extraction Script (All Datasets)
=============================================

This script extracts mono WAV audio files from videos listed in a split CSV.
It is designed to work across different datasets and handles videos with
no audio stream safely and clearly.

The script uses FFprobe first to check whether a video contains an audio
stream. If audio exists, FFmpeg is used to extract it as a 16 kHz mono WAV.
If no audio exists, the script records that status instead of failing.

This makes the pipeline more robust for datasets where some videos are
silent or contain only visual content.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
External tools required:
    - FFmpeg
    - FFprobe

You can check installation with:
    ffmpeg -version
    ffprobe -version

Python packages:
    pip install pandas tqdm

------------------------------------------------
RUN
------------------------------------------------
Example:
    python pipelines/audio/extract_audio.py --split_csv data/splits/deeperforensics_val.csv --out_dir
    data/interim/audio/DeeperForensics/val

------------------------------------------------
INPUT
------------------------------------------------
A split CSV file containing at least:
    video_path, video_id

------------------------------------------------
OUTPUTS
------------------------------------------------
For each video with audio:
    <out_dir>/<safe_id>.wav

Also writes:
    <out_dir>/_id_map.csv

The mapping file stores:
    safe_id, video_id, video_path, status, error_tail

This gives a full audit trail of what happened for every sample.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

import argparse
from pathlib import Path
import subprocess
import pandas as pd
from tqdm import tqdm
import hashlib


def make_safe_id(video_id: str) -> str:
    """
    Create a short, deterministic ID from the original video_id.

    This helps keep filenames short and Windows-safe while still
    remaining reproducible across runs.
    """
    return hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:16]


def run_cmd(cmd):
    """
    Run a shell command and capture its output.

    Returns:
        success_flag, stdout_text, stderr_text
    """
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode == 0, p.stdout, p.stderr


def has_audio_stream(video_path: str) -> bool:
    """
    Check whether a video contains an audio stream using FFprobe.

    FFprobe prints stream information if audio exists.
    If FFprobe fails for some reason, this function returns True so that
    FFmpeg still gets a chance to try extraction.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path
    ]

    ok, out, err = run_cmd(cmd)

    if not ok:
        # If ffprobe fails, treat it as unknown instead of silent.
        # This is safer because FFmpeg may still be able to extract audio.
        return True

    return len(out.strip()) > 0


def extract_wav(video_path: str, out_wav: Path):
    """
    Extract audio from a video as a mono 16 kHz WAV file.

    Output format:
        - mono channel
        - 16 kHz sampling rate
        - pcm_s16le codec

    Returns:
        success_flag, stderr_text
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", video_path,
        "-vn",              # disable video output
        "-ac", "1",         # mono audio
        "-ar", "16000",     # sample rate = 16 kHz
        "-c:a", "pcm_s16le",
        str(out_wav)
    ]

    ok, out, err = run_cmd(cmd)
    return ok, err


def main():
    """
    Main extraction workflow.

    Steps:
    1. Read the split CSV
    2. Build a safe ID for each video
    3. Check whether audio exists
    4. Extract WAV if possible
    5. Log every result to _id_map.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Audit log file: records what happened for every sample
    map_path = out_dir / "_id_map.csv"
    if not map_path.exists():
        map_path.write_text(
            "safe_id,video_id,video_path,status,error_tail\n",
            encoding="utf-8"
        )

    extracted = 0
    skipped = 0
    failed = 0
    no_audio = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting audio -> {out_dir}"):
        video_path = str(row["video_path"])
        video_id = str(row["video_id"])

        safe_id = make_safe_id(video_id)
        out_wav = out_dir / f"{safe_id}.wav"

        error_tail = ""

        # Resume-safe behavior:
        # if the WAV file already exists and is non-empty, skip extraction
        if out_wav.exists() and out_wav.stat().st_size > 0:
            skipped += 1
            status = "skipped"

        else:
            # If the video has no audio stream, record it clearly
            if not has_audio_stream(video_path):
                no_audio += 1
                status = "no_audio"

            else:
                ok, err = extract_wav(video_path, out_wav)

                if ok and out_wav.exists() and out_wav.stat().st_size > 0:
                    extracted += 1
                    status = "extracted"
                else:
                    failed += 1
                    status = "failed"

                    # Keep only the last part of stderr so logs stay compact
                    error_tail = (err[-300:].replace("\n", " ") if err else "")

                    # Print the failure to terminal for debugging
                    if error_tail:
                        print(f"\nFFmpeg failed: {video_path}\nERR_TAIL: {error_tail}\n")
                    else:
                        print(f"\nFFmpeg failed: {video_path}\n(no stderr)\n")

        # Always write one audit row so nothing is silently skipped
        with open(map_path, "a", encoding="utf-8") as f:
            f.write(f"{safe_id},{video_id},{video_path},{status},{error_tail}\n")

    print("\nDone.")
    print("Extracted:", extracted)
    print("Skipped:", skipped)
    print("No-audio:", no_audio)
    print("Failed:", failed)


if __name__ == "__main__":
    main()