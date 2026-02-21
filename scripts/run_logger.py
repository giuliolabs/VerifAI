from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import platform
import sys
import traceback
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "experiments" / "logs"

def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _safe_run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=str(ROOT), stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None

def _git_info() -> dict:
    return {
        "git_commit": _safe_run(["git", "rev-parse", "HEAD"]),
        "git_branch": _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": _safe_run(["git", "status", "--porcelain"]),
    }

def _torch_info() -> dict:
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        return {"torch_version": None}

def log_run(run_name: str, payload: dict, log_dir: Path | None = None) -> Path:
    """
    Write a VerifAI run log.
    payload recommended keys:
      - task: "train" | "eval" | "explainability" | "preprocess" | "user_study"
      - model: e.g. "xception"
      - dataset: e.g. "FaceForensics++_C23", "Celeb-DF-v2", "Survey369"
      - split: "train"/"val"/"test" or "cross_dataset"
      - checkpoint_path / config_path
      - seed
      - metrics: dict
      - artifacts: list[str]
      - notes: str
    """
    log_dir = log_dir or DEFAULT_LOG_DIR
    runs_dir = log_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

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

    out = runs_dir / f"run_{_now_stamp()}_{run_name}.json"
    out.write_text(json.dumps({"meta": meta, "payload": payload}, indent=2), encoding="utf-8")
    return out

def log_exception(run_name: str, exc: BaseException, context: dict | None = None, log_dir: Path | None = None) -> Path:
    """
    Write a VerifAI error log with traceback + context.
    """
    log_dir = log_dir or DEFAULT_LOG_DIR
    errors_dir = log_dir / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)

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

    out = errors_dir / f"error_{_now_stamp()}_{run_name}.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out
