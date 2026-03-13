"""
ml Package Entrypoint
=====================

This module acts as the main entrypoint for the `ml` package used in the
VerifAI deepfake detection framework.

Its purpose is to provide a clean and stable public interface for the
machine learning components of the project.

Goals:
- Provide a clean public API so imports remain stable and readable
- Centralize reproducibility helpers
- Expose project metadata and package utilities
- Keep the package importable even when optional dependencies are not installed

Typical usage example:

    from ml import seed_everything, get_device

    seed_everything(42)
    device = get_device()

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Used to define simple structured data containers
from dataclasses import dataclass

# Standard library utilities
import os
import random

# NumPy is used for deterministic seeding
import numpy as np


# -------------------------------------------------------
# Public API
# -------------------------------------------------------
# __all__ defines what gets imported when users run:
#
#     from ml import *
#
# This helps keep the public API clean and intentional.
__all__ = [
    "__version__",
    "ProjectInfo",
    "seed_everything",
    "get_device",
]


# -------------------------------------------------------
# Package version
# -------------------------------------------------------
# Semantic versioning for the ML package.
# This allows experiments and reports to reference
# the exact framework version used.
__version__ = "0.1.0"


# -------------------------------------------------------
# Project metadata
# -------------------------------------------------------
@dataclass(frozen=True)
class ProjectInfo:
    """
    Immutable project metadata container.

    Using a dataclass provides a simple structured way
    to expose information about the framework.
    """

    # Human-readable project name
    name: str = "Deepfake Detection Framework"

    # Python package name
    package: str = "ml"

    # Current package version
    version: str = __version__


# -------------------------------------------------------
# Reproducibility utilities
# -------------------------------------------------------
def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch.

    This improves experiment reproducibility so that
    training and evaluation runs produce consistent results.

    Args:
        seed:
            Random seed value to use.

        deterministic:
            If True, enables deterministic PyTorch behavior.
            This may reduce GPU performance but increases
            repeatability between runs.
    """

    # Python hash seed (affects hashing order in dictionaries)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python built-in random module
    random.seed(seed)

    # NumPy random number generator
    np.random.seed(seed)

    # -------------------------------------------------------
    # Optional PyTorch seeding
    # -------------------------------------------------------
    # PyTorch is not required for importing this package,
    # so we wrap it in a try/except block.
    try:
        import torch

        # Seed CPU RNG
        torch.manual_seed(seed)

        # Seed all CUDA devices
        torch.cuda.manual_seed_all(seed)

        # Optional deterministic settings
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    except Exception:
        # If PyTorch is not installed, silently skip it.
        # This keeps the package usable in environments
        # where only preprocessing or evaluation is needed.
        pass


# -------------------------------------------------------
# Device selection helper
# -------------------------------------------------------
def get_device(prefer_gpu: bool = True) -> str:
    """
    Determine which compute device should be used.

    Returns:
        "cuda" if a GPU is available and prefer_gpu=True,
        otherwise "cpu".

    The function safely falls back to CPU if PyTorch is
    not installed or GPU support is unavailable.

    Args:
        prefer_gpu:
            If False, always return CPU even if GPU exists.
    """

    # If user explicitly disables GPU usage
    if not prefer_gpu:
        return "cpu"

    try:
        import torch

        # Return CUDA if available, otherwise CPU
        return "cuda" if torch.cuda.is_available() else "cpu"

    except Exception:
        # If torch is not installed, default to CPU
        return "cpu"