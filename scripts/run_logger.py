"""
VerifAI Run & Error Logging Utilities
=====================================

This module provides reusable utilities for logging experiment runs and
exceptions in the VerifAI deepfake detection framework.

The goal is to improve:
- experiment reproducibility
- debugging traceability
- project transparency for examiners

The module records useful metadata such as:
- timestamp
- Python and platform versions
- Git commit / branch
- PyTorch / CUDA environment
- experiment parameters and metrics

Logs are written into:

    experiments/logs/runs/
    experiments/logs/errors/

Each run or error is saved as a JSON file so results can be inspected,
reproduced, or analysed later.

Typical usage:

    from scripts.log_utils import log_run, log_exception

    log_run(
        run_name="xception_eval",
        payload={"dataset": "FaceForensics++", "metrics": {...}}
    )

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# -------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------

# JSON is used to serialize run logs into structured files
import json

# pathlib allows platform-independent path handling
from pathlib import Path

# datetime generates timestamps for run identifiers
from datetime import datetime

# platform and sys capture environment information
import platform
import sys

# traceback captures detailed exception stack traces
import traceback

# subprocess allows us to query git information
import subprocess


# -------------------------------------------------------
# Project root and default logging directory
# -------------------------------------------------------

# ROOT refers to the project root directory.
# This file lives in /scripts, so we go one level up.
ROOT = Path(__file__).resolve().parents[1]

# Default location for all experiment logs
DEFAULT_LOG_DIR = ROOT / "experiments" / "logs"


# -------------------------------------------------------
# Helper: timestamp generator
# -------------------------------------------------------
def _now_stamp() -> str:
    """
    Generate a filesystem-safe timestamp.

    Example output:
        2026-03-15_14-32-05
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# -------------------------------------------------------
# Helper: run shell commands safely
# -------------------------------------------------------
def _safe_run(cmd: list[str]) -> str | None:
    """
    Execute a command safely and return its output.

    If the command fails (e.g., git not installed),
    return None instead of raising an exception.
    """
    try:
        out = subprocess.check_output(
            cmd,
            cwd=str(ROOT),
            stderr=subprocess.STDOUT,
            text=True,
        )
        return out.strip()

    except Exception:
        return None


# -------------------------------------------------------
# Helper: collect Git metadata
# -------------------------------------------------------
def _git_info() -> dict:
    """
    Retrieve basic Git repository metadata.

    This helps track exactly which code version
    produced a given experiment result.
    """
    return {
        "git_commit": _safe_run(["git", "rev-parse", "HEAD"]),
        "git_branch": _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": _safe_run(["git", "status", "--porcelain"]),
    }


# -------------------------------------------------------
# Helper: collect PyTorch environment info
# -------------------------------------------------------
def _torch_info() -> dict:
    """
    Retrieve information about the PyTorch runtime.

    Includes GPU availability and CUDA details if present.
    """
    try:
        import torch

        return {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }

    except Exception:
        # If PyTorch is not installed, return minimal info
        return {"torch_version": None}


# -------------------------------------------------------
# Run logging function
# -------------------------------------------------------
def log_run(
    run_name: str,
    payload: dict,
    log_dir: Path | None = None,
) -> Path:
    """
    Write a VerifAI experiment run log.

    The payload dictionary should contain the
    experiment configuration and results.

    Recommended payload keys:

        task:
            "train" | "eval" | "explainability" | "preprocess" | "user_study"

        model:
            e.g. "xception", "vit", "multimodal_fusion"

        dataset:
            e.g. "FaceForensics++_C23", "Celeb-DF-v2", "Survey369"

        split:
            "train" | "val" | "test" | "cross_dataset"

        checkpoint_path:
            model checkpoint used

        config_path:
            training configuration file

        seed:
            random seed used for reproducibility

        metrics:
            dictionary of evaluation results

        artifacts:
            list of generated output files

        notes:
            optional experiment notes

    Returns:
        Path to the generated JSON run log file.
    """

    # Use default logs directory if none is provided
    log_dir = log_dir or DEFAULT_LOG_DIR

    # Create runs directory if missing
    runs_dir = log_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Collect environment metadata
    # -------------------------------------------------------
    meta = {
        "project": "VerifAI",
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        **_git_info(),
        **_torch_info(),
    }

    # Construct output filename
    out = runs_dir / f"run_{_now_stamp()}_{run_name}.json"

    # Write JSON log
    out.write_text(
        json.dumps({"meta": meta, "payload": payload}, indent=2),
        encoding="utf-8",
    )

    return out


# -------------------------------------------------------
# Exception logging function
# -------------------------------------------------------
def log_exception(
    run_name: str,
    exc: BaseException,
    context: dict | None = None,
    log_dir: Path | None = None,
) -> Path:
    """
    Write a structured error log for debugging.

    The log captures:
    - exception type
    - exception message
    - full traceback
    - optional contextual information

    This allows easier debugging of failed runs.

    Args:
        run_name:
            identifier for the experiment or task

        exc:
            the exception object that occurred

        context:
            optional dictionary containing additional
            debugging information

    Returns:
        Path to the saved error JSON file.
        :param run_name:
        :param exc:
        :param context:
        :param log_dir:
    """

    # Use default logs directory if not specified
    log_dir = log_dir or DEFAULT_LOG_DIR

    # Create errors directory
    errors_dir = log_dir / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)

    # Build error payload
    data = {
        "project": "VerifAI",
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "context": context or {},
        **_git_info(),
        **_torch_info(),
    }

    # Output file path
    out = errors_dir / f"error_{_now_stamp()}_{run_name}.json"

    # Save error information
    out.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    return out