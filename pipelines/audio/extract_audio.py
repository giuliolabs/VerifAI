"""
Generic Audio Extraction Script (All Datasets) - robust for videos with NO audio.

Dependencies:
    - FFmpeg + FFprobe (installed via winget)
        ffmpeg -version
        ffprobe -version
    pip install pandas tqdm

Run:
    python pipelines/audio/extract_audio.py --split_csv data/splits/deeperforensics_val.csv --out_dir data/interim/audio/DeeperForensics/val

Outputs:
    <out_dir>/<safe_id>.wav
    <out_dir>/_id_map.csv
"""

import argparse
from pathlib import Path
import subprocess
import pandas as pd
from tqdm import tqdm
import hashlib


def make_safe_id(video_id: str) -> str:
    return hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:16]


def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode == 0, p.stdout, p.stderr


def has_audio_stream(video_path: str) -> bool:
    # ffprobe returns stream lines if audio exists, otherwise empty
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
        # if ffprobe fails, treat as "unknown" -> let ffmpeg try
        return True
    return len(out.strip()) > 0


def extract_wav(video_path: str, out_wav: Path):
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_wav)
    ]
    ok, out, err = run_cmd(cmd)
    return ok, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_path = out_dir / "_id_map.csv"
    if not map_path.exists():
        map_path.write_text("safe_id,video_id,video_path\n", encoding="utf-8")

    extracted = skipped = failed = no_audio = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting audio -> {out_dir}"):
        video_path = str(row["video_path"])
        video_id = str(row["video_id"])

        safe_id = make_safe_id(video_id)
        out_wav = out_dir / f"{safe_id}.wav"

        # resume-safe
        if out_wav.exists() and out_wav.stat().st_size > 0:
            skipped += 1
            continue

        # NEW: skip silent videos cleanly
        if not has_audio_stream(video_path):
            no_audio += 1
            continue

        ok, err = extract_wav(video_path, out_wav)

        if ok and out_wav.exists() and out_wav.stat().st_size > 0:
            extracted += 1
            with open(map_path, "a", encoding="utf-8") as f:
                f.write(f"{safe_id},{video_id},{video_path}\n")
        else:
            failed += 1
            # keep stderr short (tail is most useful)
            err_tail = err[-300:].replace("\n", " ")
            print(f"\nFFmpeg failed: {video_path}\nERR_TAIL: {err_tail}\n")

    print("\nDone.")
    print("Extracted:", extracted)
    print("Skipped:", skipped)
    print("No-audio:", no_audio)
    print("Failed:", failed)


if __name__ == "__main__":
    main()
