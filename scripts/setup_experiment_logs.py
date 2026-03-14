"""
VerifAI Logging Scaffold Initializer
====================================

This script sets up the central logging structure used across the whole
VerifAI project. Its purpose is to create a consistent place for storing
debug notes, reproducibility records, run metadata, error reports, and
training curves.

The script is useful because it keeps experiment evidence organised in
one place, which supports:
- reproducibility
- debugging
- clearer project structure
- stronger evidence for examiners

What this script creates:
- experiments/logs/debug_log.md
- experiments/logs/verifai_random_seeds.tsv
- experiments/logs/runs/
- experiments/logs/errors/
- experiments/logs/training_curves/
- experiments/logs/README.md

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Path is used for platform-independent file and folder handling
from pathlib import Path

# datetime is used to print a timestamp when setup finishes
from datetime import datetime


# Repo root is the parent folder above /scripts.
# This makes the script work relative to the project structure.
ROOT = Path(__file__).resolve().parents[1]

# Central logs directory for the whole project
# (not limited to one experiment type)
LOGS_DIR = ROOT / "experiments" / "logs"


# Template text for the main debugging diary.
# This gives the project a ready-made structured log format
# instead of leaving debugging notes unorganized.
DEBUG_TEMPLATE = """# VerifAI Debug / Error Tracking Log

This log documents bugs, investigation steps, and fixes across VerifAI:
data ingestion, preprocessing, training, evaluation, API, explainability, and user study.
Keeping this log improves reproducibility and demonstrates rigour for examiners.

---

## Common issues seen in this project

- Missing/invalid file paths (dataset folder names differ, OneDrive sync issues)
- CSV split mismatch (empty test CSV, wrong root paths)
- Torch dtype mismatch (Double vs Float) during eval
- Checkpoint compatibility (missing/unexpected keys; timm xception -> legacy_xception warning)
- Large archive extraction issues (tar/zip join problems, memory limits)
- Frame/audio extraction failures for particular videos (corrupt or unsupported codec)

---

### Date: 21/02/2026
### Component: Backend API (Render deployment)

### Context: 
Deploy VerifAI FastAPI backend on Render and run /api/predict (multimodal inference: video frames + 
MFCC audio) using the checkpoint experiments/results/fakeavceleb_av_fusion_v1/best_model.pt. Tested via UI and 
PowerShell curl.exe uploads.

### Symptom / Error: 
1. UI returned HTTP 502 during upload/inference.
2. Render sent email: “instance exceeded its memory limit, triggered automatic restart.”
3. Logs showed routing errors during sleep/cold start (e.g., x-render-routing: dynamic-hibernate-error-503) and requests 
failing while the service restarted.

### Investigation:
1. Verified DNS routing (initially inconsistent due to local DNS resolver/router cache; fixed by switching Windows DNS 
to Google DNS).
2. Tested API directly with PowerShell curl.exe including X-API-Key.
3. Confirmed redirects between apex and www domains and fixed tests by targeting the canonical host.
4. Checked Render logs/alerts and correlated 502s with memory-limit restarts during inference.

### Root Cause:
Render free-tier resources (~512MB RAM, very low CPU) are insufficient for multimodal PyTorch inference + video decoding
 + audio MFCC extraction. Peak RAM spikes during preprocessing/inference cause the process to be killed (OOM), leading 
 to 502 and automatic restarts.

### Fix:
1. Operational mitigations: use small test clips, warm the service before requests, force single worker to avoid 
multiple model copies.
2. Code mitigations (planned/implemented): reduce peak memory (CPU-only execution, limit torch threads, 
torch.inference_mode(), explicit tensor cleanup).
3. Deployment fix (planned): upgrade instance (target 2GB RAM / 1 CPU) to run inference reliably.

### Verification:
1. Confirmed correct domain routing and API access via curl.exe (no Apache/Bluehost interference).
2. Verified API-key enforcement (401 without key, accepted with key).
3. After DNS fix, requests consistently reached Render/uvicorn; remaining failures aligned with OOM restarts during 
inference.

### Notes:
1. Free-tier hibernation can produce transient 503s; repeated requests after wake-up are expected.
2. For stable demo: upgrade resources or simplify preprocessing (fewer frames, smaller resize, optional audio 
disable) to keep inference within RAM limits.

---
"""


def main() -> None:
    """
    Create the central VerifAI logging folder structure and starter files.

    This function:
    - creates the main logs directory
    - creates a debug diary if it does not already exist
    - creates a random-seeds registry file
    - creates subfolders for run logs, error logs, and training curves
    - creates a README explaining the purpose of the logs folder
    """

    # Ensure the main logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Debug diary
    # -------------------------------------------------------
    # This file stores structured debugging notes and investigation history.
    debug_log = LOGS_DIR / "debug_log.md"
    if not debug_log.exists():
        debug_log.write_text(DEBUG_TEMPLATE, encoding="utf-8")

    # -------------------------------------------------------
    # Seed registry
    # -------------------------------------------------------
    # This TSV file can be used to record random seeds used in experiments,
    # which helps make training runs more reproducible.
    seeds_file = LOGS_DIR / "verifai_random_seeds.tsv"
    if not seeds_file.exists():
        seeds_file.write_text(
            "seed\tnote\ttimestamp\n",
            encoding="utf-8",
        )

    # -------------------------------------------------------
    # Run logs and error logs
    # -------------------------------------------------------
    # These folders keep JSON outputs separate by purpose:
    # - runs/ for normal experiment metadata
    # - errors/ for exceptions and failure reports
    (LOGS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "errors").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Training curves
    # -------------------------------------------------------
    # This folder stores per-run CSV logs and possibly curve plots later on.
    (LOGS_DIR / "training_curves").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Logs README
    # -------------------------------------------------------
    # This file explains the purpose of the logging directory
    # so the structure stays understandable for examiners and collaborators.
    readme = LOGS_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            "# VerifAI Experiment Logs\n\n"
            "Central location for reproducibility artifacts:\n\n"
            "- `debug_log.md`: structured debugging diary\n"
            "- `verifai_random_seeds.tsv`: seeds registry\n"
            "- `runs/`: JSON run metadata (configs, checkpoints, datasets, metrics)\n"
            "- `errors/`: JSON exception logs with tracebacks\n"
            "- `training_curves/`: per-run CSV curves + generated plots\n",
            encoding="utf-8",
        )

    # Add a timestamp to the final confirmation message
    # so it is clear when the scaffold was created or checked.
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[OK] VerifAI logs scaffold ready at: {LOGS_DIR} ({stamp})")


if __name__ == "__main__":
    main()