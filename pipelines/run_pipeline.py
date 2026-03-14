"""
VerifAI Pipeline Orchestrator
============================

This script provides one central command for running the full VerifAI
preprocessing pipeline. Instead of manually launching each preprocessing
script one by one, this orchestrator calls them in the correct order.

Main pipeline steps supported:
- build dataset manifests
- build train/validation/test splits
- run integrity checks
- extract video frames
- extract audio
- compute audio features
- compute video features
- compute metadata features

The script intentionally uses subprocess calls to the existing scripts
already present in the repository. This keeps the workflow aligned with
the rest of the project and makes the process easier for examiners to
follow and reproduce.

Usage examples:
--------------
python pipelines/run_pipeline.py --datasets all --stages all

python pipelines/run_pipeline.py --datasets celebdfv2 faceforensics++_c23 --stages splits integrity frames
audio audio_features video_features metadata_features

python pipelines/run_pipeline.py --datasets all --stages all --frames_per_video 5 --n_mfcc 40

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


# -------------------------
# Dataset configuration
# -------------------------
@dataclass(frozen=True)
class DatasetCfg:
    """
    Stores all important information needed to run the pipeline
    for one dataset.
    """
    key: str                 # canonical dataset key used by this orchestrator
    display_name: str        # actual folder name used under data/interim and data/processed
    manifest_builder: str    # script path used to build the dataset manifest
    split_train_csv: str
    split_val_csv: str
    split_test_csv: str


# IMPORTANT:
# These split CSV filenames must match the real files used in data/splits.
DATASETS: Dict[str, DatasetCfg] = {
    "celebdfv2": DatasetCfg(
        key="celebdfv2",
        display_name="Celeb-DF-v2",
        manifest_builder="pipelines/splits/build_celebdfv2_manifest.py",
        split_train_csv="data/splits/celebdfv2_train.csv",
        split_val_csv="data/splits/celebdfv2_val.csv",
        split_test_csv="data/splits/celebdfv2_test.csv",
    ),
    "deeperforensics": DatasetCfg(
        key="deeperforensics",
        display_name="DeeperForensics",
        manifest_builder="pipelines/splits/build_deeperforensics_manifest.py",
        split_train_csv="data/splits/deeperforensics_train.csv",
        split_val_csv="data/splits/deeperforensics_val.csv",
        split_test_csv="data/splits/deeperforensics_test.csv",
    ),
    "faceforensics++_c23": DatasetCfg(
        key="faceforensics++_c23",
        display_name="FaceForensics++_C23",
        manifest_builder="pipelines/splits/build_faceforensics++_c23_manifest.py",
        split_train_csv="data/splits/faceforensics++_c23_train.csv",
        split_val_csv="data/splits/faceforensics++_c23_val.csv",
        split_test_csv="data/splits/faceforensics++_c23_test.csv",
    ),
    "fakeavceleb": DatasetCfg(
        key="fakeavceleb",
        display_name="FakeAVCeleb_v1.2",
        manifest_builder="pipelines/splits/build_fakeavceleb_manifest.py",
        split_train_csv="data/splits/fakeavceleb_train.csv",
        split_val_csv="data/splits/fakeavceleb_val.csv",
        split_test_csv="data/splits/fakeavceleb_test.csv",
    ),
}


# -------------------------
# Script paths in the repo
# -------------------------
# These are the individual scripts called by the orchestrator.
MAKE_SPLITS = "pipelines/splits/make_splits.py"
CHECK_INTEGRITY = "pipelines/validate/check_integrity.py"

EXTRACT_FRAMES_FROM_CSV = "pipelines/video/extract_frames_from_csv.py"
EXTRACT_AUDIO = "pipelines/audio/extract_audio.py"

AUDIO_FEATURES = "pipelines/audio/audio_features.py"
VIDEO_FEATURES = "pipelines/video/video_features.py"
METADATA_FEATURES = "pipelines/metadata/extract_metadata.py"


# -------------------------
# Helper functions
# -------------------------
def run(cmd: List[str], cwd: Path | None = None) -> None:
    """
    Run a subprocess command and stop the pipeline if it fails.

    This keeps each step explicit and easy to debug.
    """
    print("\n>>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def assert_exists(path: str, what: str) -> None:
    """
    Check that an expected file or script exists before using it.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected {what} not found: {p.as_posix()}")


def split_paths(cfg: DatasetCfg) -> List[str]:
    """
    Return all split CSV paths for one dataset.
    """
    return [cfg.split_train_csv, cfg.split_val_csv, cfg.split_test_csv]


def frames_out_dir(cfg: DatasetCfg, split: str) -> str:
    """
    Output folder for extracted frames for one dataset split.
    """
    return f"data/interim/frames/{cfg.display_name}/{split}"


def audio_out_dir(cfg: DatasetCfg, split: str) -> str:
    """
    Output folder for extracted audio for one dataset split.
    """
    return f"data/interim/audio/{cfg.display_name}/{split}"


def audio_features_out_dir(cfg: DatasetCfg, split: str) -> str:
    """
    Output folder for processed audio features for one dataset split.
    """
    return f"data/processed/audio_features/{cfg.display_name}/{split}"


def video_features_out_dir(cfg: DatasetCfg, split: str) -> str:
    """
    Output folder for processed video features for one dataset split.
    """
    return f"data/processed/video_features/{cfg.display_name}/{split}"


def metadata_features_out_dir(cfg: DatasetCfg, split: str) -> str:
    """
    Output folder for metadata-based features for one dataset split.
    """
    return f"data/processed/metadata_features/{cfg.display_name}/{split}"


# -------------------------
# Pipeline stages
# -------------------------
def stage_manifests(datasets: List[DatasetCfg]) -> None:
    """
    Build manifests for each selected dataset.
    """
    for cfg in datasets:
        assert_exists(cfg.manifest_builder, "manifest builder script")
        run([sys.executable, cfg.manifest_builder])


def stage_splits() -> None:
    """
    Build train/validation/test splits.

    This stage is run once because make_splits.py usually handles
    all manifests together.
    """
    assert_exists(MAKE_SPLITS, "make_splits.py")
    run([sys.executable, MAKE_SPLITS])


def stage_integrity() -> None:
    """
    Run project-wide data integrity checks.
    """
    assert_exists(CHECK_INTEGRITY, "check_integrity.py")
    run([sys.executable, CHECK_INTEGRITY])


def stage_frames(datasets: List[DatasetCfg], frames_per_video: int) -> None:
    """
    Extract video frames for each selected dataset and split.
    """
    assert_exists(EXTRACT_FRAMES_FROM_CSV, "extract_frames_from_csv.py")

    for cfg in datasets:
        for split_name, split_csv in [
            ("train", cfg.split_train_csv),
            ("val", cfg.split_val_csv),
            ("test", cfg.split_test_csv),
        ]:
            assert_exists(split_csv, f"{cfg.key} {split_name} split CSV")
            out_dir = frames_out_dir(cfg, split_name)

            run([
                sys.executable, EXTRACT_FRAMES_FROM_CSV,
                "--split_csv", split_csv,
                "--out_dir", out_dir,
                "--frames_per_video", str(frames_per_video),
            ])


def stage_audio(datasets: List[DatasetCfg]) -> None:
    """
    Extract audio waveforms for each selected dataset and split.
    """
    assert_exists(EXTRACT_AUDIO, "extract_audio.py")

    for cfg in datasets:
        for split_name, split_csv in [
            ("train", cfg.split_train_csv),
            ("val", cfg.split_val_csv),
            ("test", cfg.split_test_csv),
        ]:
            assert_exists(split_csv, f"{cfg.key} {split_name} split CSV")
            out_dir = audio_out_dir(cfg, split_name)

            run([
                sys.executable, EXTRACT_AUDIO,
                "--split_csv", split_csv,
                "--out_dir", out_dir,
            ])


def stage_audio_features(datasets: List[DatasetCfg], n_mfcc: int) -> None:
    """
    Compute MFCC or other audio features from previously extracted WAV files.
    """
    assert_exists(AUDIO_FEATURES, "audio_features.py")

    for cfg in datasets:
        for split_name in ["train", "val", "test"]:
            wav_dir = audio_out_dir(cfg, split_name)
            out_dir = audio_features_out_dir(cfg, split_name)

            # Skip safely if audio extraction has not been run yet
            if not Path(wav_dir).exists():
                print(f"\n[SKIP] WAV dir not found (no audio extracted?): {wav_dir}")
                continue

            run([
                sys.executable, AUDIO_FEATURES,
                "--wav_dir", wav_dir,
                "--out_dir", out_dir,
                "--n_mfcc", str(n_mfcc),
            ])


def stage_video_features(datasets: List[DatasetCfg]) -> None:
    """
    Compute video-level processed features from extracted frames.
    """
    assert_exists(VIDEO_FEATURES, "video_features.py")

    for cfg in datasets:
        for split_name in ["train", "val", "test"]:
            frames_dir = frames_out_dir(cfg, split_name)
            out_dir = video_features_out_dir(cfg, split_name)

            # Skip safely if frame extraction has not been run yet
            if not Path(frames_dir).exists():
                print(f"\n[SKIP] Frames dir not found (run frames stage first?): {frames_dir}")
                continue

            run([
                sys.executable, VIDEO_FEATURES,
                "--frames_dir", frames_dir,
                "--out_dir", out_dir,
            ])


def stage_metadata_features(datasets: List[DatasetCfg]) -> None:
    """
    Compute metadata-based features from split CSV files.
    """
    assert_exists(METADATA_FEATURES, "extract_metadata.py")

    for cfg in datasets:
        for split_name, split_csv in [
            ("train", cfg.split_train_csv),
            ("val", cfg.split_val_csv),
            ("test", cfg.split_test_csv),
        ]:
            if not Path(split_csv).exists():
                print(f"\n[SKIP] Split CSV not found: {split_csv}")
                continue

            out_dir = metadata_features_out_dir(cfg, split_name)

            run([
                sys.executable, METADATA_FEATURES,
                "--split_csv", split_csv,
                "--out_dir", out_dir,
            ])


# -------------------------
# Main
# -------------------------
def main() -> None:
    """
    Parse user arguments, resolve selected datasets/stages,
    and run the pipeline in the expected order.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to run: all | celebdfv2 deeperforensics faceforensics++_c23 fakeavceleb"
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help="Stages: all | manifests splits integrity frames audio audio_features video_features metadata_features"
    )

    parser.add_argument("--frames_per_video", type=int, default=5)
    parser.add_argument("--n_mfcc", type=int, default=40)

    args = parser.parse_args()

    # Resolve selected datasets
    if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        selected = list(DATASETS.values())
    else:
        selected = []
        for key in args.datasets:
            k = key.lower()
            if k not in DATASETS:
                raise ValueError(f"Unknown dataset key: {key}. Valid: {list(DATASETS.keys())} or 'all'")
            selected.append(DATASETS[k])

    # Resolve selected stages
    stages = [s.lower() for s in args.stages]
    if len(stages) == 1 and stages[0] == "all":
        stages = [
            "manifests",
            "splits",
            "integrity",
            "frames",
            "audio",
            "audio_features",
            "video_features",
            "metadata_features",
        ]

    print("\n=== VerifAI Orchestrator ===")
    print("Datasets:", [d.key for d in selected])
    print("Stages:", stages)
    print("frames_per_video:", args.frames_per_video, "n_mfcc:", args.n_mfcc)

    # Execute stages in the correct order
    if "manifests" in stages:
        stage_manifests(selected)

    if "splits" in stages:
        stage_splits()

        # After making splits, check that the expected files now exist
        for cfg in selected:
            for sp in split_paths(cfg):
                assert_exists(sp, f"split CSV for {cfg.key}")

    if "integrity" in stages:
        stage_integrity()

    if "frames" in stages:
        stage_frames(selected, frames_per_video=args.frames_per_video)

    if "audio" in stages:
        stage_audio(selected)

    if "audio_features" in stages:
        stage_audio_features(selected, n_mfcc=args.n_mfcc)

    if "video_features" in stages:
        stage_video_features(selected)

    if "metadata_features" in stages:
        stage_metadata_features(selected)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()