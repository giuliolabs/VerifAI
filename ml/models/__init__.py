"""
Model Registry + Factory (VerifAI)
===================================

This module implements a centralized model registry and factory
pattern aligned with the VerifAI repository structure.

Purpose:
- Provide a single source of truth for available models
- Enable dynamic model creation from string identifiers
- Cleanly separate model definition from model instantiation logic
- Improve scalability and experimentation workflow

Each model entry defines:
- name
- task type (video | audio | fusion)
- builder_path ("module.path:callable")
- description

This design supports flexible experimentation without hardcoding
imports across the codebase.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# dataclass used to define immutable model metadata
from dataclasses import dataclass

# typing used for clarity and safer interfaces
from typing import Any, Callable, Dict, Optional, Tuple


# Public API of this module
__all__ = ["ModelSpec", "list_models", "get_model_spec", "create_model"]

# Builder represents any callable that returns a model instance
Builder = Callable[..., Any]


# ------------------------------------------------
# Model Specification
# ------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """
    Immutable metadata describing a model entry in the registry.

    name:
        Unique string identifier used in experiments.

    task:
        Category of model ("video", "audio", "fusion").

    builder_path:
        Import path to builder function in the format:
        "module.path:callable_name"

    description:
        Short human-readable explanation.
    """
    name: str
    task: str              # "video" | "audio" | "fusion"
    builder_path: str      # "module.path:callable"
    description: str = ""


# ------------------------------------------------
# Registry Definition
# ------------------------------------------------

# Central registry mapping model names to specifications.
# Adding a new model only requires inserting one new entry here.
_REGISTRY: Dict[str, ModelSpec] = {

    # ---- Video Models ----
    "mobilenet_baseline": ModelSpec(
        name="mobilenet_baseline",
        task="video",
        builder_path="ml.models.video.mobilenet_baseline:build_model",
        description="Lightweight CNN baseline (good for speed + ablations).",
    ),

    "xception": ModelSpec(
        name="xception",
        task="video",
        builder_path="ml.models.video.xception:build_model",
        description="Strong CNN baseline widely used in deepfake detection.",
    ),

    "vit": ModelSpec(
        name="vit",
        task="video",
        builder_path="ml.models.video.vit:build_model",
        description="Vision Transformer baseline.",
    ),

    "temporal_transformer": ModelSpec(
        name="temporal_transformer",
        task="video",
        builder_path="ml.models.video.temporal_transformer:build_model",
        description="Sequence model for video-level reasoning over frame embeddings.",
    ),

    # ---- Audio Models ----
    "mfcc_cnn": ModelSpec(
        name="mfcc_cnn",
        task="audio",
        builder_path="ml.models.audio.mfcc_cnn:build_model",
        description="CNN over MFCC/log-mel style features (audio deepfake baseline).",
    ),

    "wav_encoder": ModelSpec(
        name="wav_encoder",
        task="audio",
        builder_path="ml.models.audio.wav_encoder:build_model",
        description="Audio encoder model (raw waveform or learned frontend).",
    ),

    # ---- Fusion Models ----
    "multimodal_fusion": ModelSpec(
        name="multimodal_fusion",
        task="fusion",
        builder_path="ml.models.fusion.multimodal_fusion:build_model",
        description="Audio-visual fusion model combining audio + video streams.",
    ),
}


# ------------------------------------------------
# Registry Interface Functions
# ------------------------------------------------

def list_models(task: Optional[str] = None) -> Tuple[str, ...]:
    """
    List available models.

    Args:
        task: optionally filter by task type ("video", "audio", "fusion")

    Returns:
        Tuple of model names (sorted).
    """

    if task is None:
        return tuple(sorted(_REGISTRY.keys()))

    return tuple(
        sorted([n for n, s in _REGISTRY.items() if s.task == task])
    )


def get_model_spec(name: str) -> ModelSpec:
    """
    Retrieve ModelSpec metadata by name.
    """

    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {', '.join(list_models())}"
        )

    return _REGISTRY[name]


# ------------------------------------------------
# Dynamic Import Utility
# ------------------------------------------------

def _import_from_path(path: str) -> Builder:
    """
    Dynamically import a builder function from a string path.

    Expected format:
        "module.path:callable_name"

    This allows decoupling the registry from direct imports,
    improving modularity and preventing circular dependencies.
    """

    if ":" not in path:
        raise ValueError(
            f"Invalid builder_path '{path}'. Expected 'module:callable'."
        )

    module_path, attr = path.split(":", 1)

    # Import module dynamically
    module = __import__(module_path, fromlist=[attr])

    # Retrieve callable
    fn = getattr(module, attr, None)

    if fn is None:
        raise ImportError(
            f"Could not find '{attr}' inside '{module_path}'."
        )

    if not callable(fn):
        raise TypeError(
            f"Imported '{module_path}:{attr}' is not callable."
        )

    return fn


# ------------------------------------------------
# Factory Function
# ------------------------------------------------

def create_model(name: str, **kwargs: Any) -> Any:
    """
    Instantiate a model from the registry.

    Args:
        name: model identifier
        **kwargs: arguments passed to the model builder

    Returns:
        Model instance
    """

    spec = get_model_spec(name)

    # Dynamically import the builder function
    builder = _import_from_path(spec.builder_path)

    # Instantiate model with provided arguments
    return builder(**kwargs)