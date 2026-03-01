"""
ml package entrypoint.

Goals:
- Give the project a clean public API (imports become stable and readable)
- Centralize reproducibility helpers (seeding, device)
- Expose the model registry (ml.models) in a single place
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random

import numpy as np


__all__ = [
    "__version__",
    "ProjectInfo",
    "seed_everything",
    "get_device",
]

__version__ = "0.1.0"


@dataclass(frozen=True)
class ProjectInfo:
    name: str = "Deepfake Detection Framework"
    package: str = "ml"
    version: str = __version__


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set seeds for reproducibility.

    deterministic=True can reduce throughput but improves repeatability.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch is optional: if installed, seed it too
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Keep package importable even without torch installed
        pass


def get_device(prefer_gpu: bool = True) -> str:
    """
    Returns 'cuda' if available (and prefer_gpu), otherwise 'cpu'.
    Uses torch if available; otherwise defaults to cpu.
    """
    if not prefer_gpu:
        return "cpu"

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
