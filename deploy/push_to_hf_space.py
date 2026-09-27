"""
Deploy the VerifAI API to a free Hugging Face Space.
====================================================

Uploads ONLY what the inference server needs (backend code, model
definitions and the three final checkpoints, ~130 MB) to a Docker Space.
Large files are handled automatically by huggingface_hub (no git-lfs setup).

Usage (from the repository root):

    pip install -U huggingface_hub
    huggingface-cli login            # paste a WRITE token from hf.co/settings/tokens
    python deploy/push_to_hf_space.py --space <hf-username>/verifai

    # preview the file list without uploading:
    python deploy/push_to_hf_space.py --space <hf-username>/verifai --dry-run

The API will then be live at:  https://<hf-username>-verifai.hf.space

Author: Giulio Dajani
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = Path(__file__).resolve().parent / "hf-space"

# Everything the running API imports or loads
INCLUDE_DIRS = ["backend", "ml"]
INCLUDE_FILES = [
    "experiments/results/final_visual_all_datasets_xception/best_model.pt",
    "experiments/results/fakeavceleb_audio_resnet_baseline/best_model.pt",
    "experiments/results/final_hybrid_av/fusion_model.joblib",
    "experiments/results/final_hybrid_av/fusion_meta.json",
    "LICENSE.md",
]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".sqlite"}


def collect() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for d in INCLUDE_DIRS:
        for p in sorted((REPO_ROOT / d).rglob("*")):
            if p.is_file() and not (set(p.parts) & SKIP_PARTS) and p.suffix not in SKIP_SUFFIXES:
                files.append((p, p.relative_to(REPO_ROOT).as_posix()))
    for f in INCLUDE_FILES:
        p = REPO_ROOT / f
        if not p.exists():
            raise SystemExit(f"Missing required file: {f}")
        files.append((p, f))
    # Space-specific files go to the Space root (Dockerfile, README card, requirements)
    for p in sorted(SPACE_DIR.iterdir()):
        files.append((p, p.name))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="e.g. giuliolabs/verifai")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = collect()
    total = sum(p.stat().st_size for p, _ in files)
    print(f"{len(files)} files, {total / 1e6:.1f} MB -> spaces/{args.space}")
    if args.dry_run:
        for p, dest in files:
            print(f"  {p.stat().st_size / 1e6:8.2f} MB  {dest}")
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p)) for p, dest in files]
    api.create_commit(
        repo_id=args.space,
        repo_type="space",
        operations=ops,
        commit_message="Deploy VerifAI API",
    )
    owner, name = args.space.split("/")
    sub = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
    print("\nUploaded. The Space is now building (first build ~5-10 min).")
    print(f"  Build logs : https://huggingface.co/spaces/{args.space}")
    print(f"  API URL    : https://{sub}.hf.space")
    print(f"  Health     : https://{sub}.hf.space/api/health")
    print("\nPut that API URL into website/config.js, then push to GitHub.")


if __name__ == "__main__":
    main()
