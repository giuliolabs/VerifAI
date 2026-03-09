"""
Video Model Registry – VerifAI
==============================

This module provides a centralized registry for all video-based
deepfake detection models used within the VerifAI framework.

It enables clean, consistent model instantiation using a simple
string identifier instead of hardcoding imports throughout the codebase.

Example usage:
    from ml.models.video import get_video_model_builder
    builder = get_video_model_builder("xception")
    model = builder(...)

Registered models:
- xception
- mobilenetv2
- vit
- temporal (Temporal Transformer)

This design improves modularity, scalability, and experimentation.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Callable allows typing for builder functions
# Dict is used to define the registry mapping
from typing import Callable, Dict

# Import available model builder functions
from .xception import build_xception_binary
from .mobilenet_baseline import build_mobilenet_v2_binary
from .vit import build_vit_binary
from .temporal_transformer import build_temporal_transformer_model


# ------------------------------------------------
# Registry Mapping
# ------------------------------------------------

# Dictionary mapping model names (string keys)
# to their corresponding builder functions.
#
# This avoids long conditional chains (if/elif)
# and makes adding new models straightforward.
VIDEO_MODEL_BUILDERS: Dict[str, Callable[..., object]] = {
    "xception": build_xception_binary,
    "mobilenetv2": build_mobilenet_v2_binary,
    "vit": build_vit_binary,
    "temporal": build_temporal_transformer_model,
}


def get_video_model_builder(name: str):
    """
    Retrieve a model builder function by name.

    Args:
        name: string identifier of the video model

    Returns:
        Callable that builds the requested model

    Raises:
        KeyError if the model name is not registered.
    """

    # Normalize input to avoid case-sensitivity issues
    key = name.strip().lower()

    # Validate that the requested model exists in the registry
    if key not in VIDEO_MODEL_BUILDERS:
        raise KeyError(
            f"Unknown video model: {name}. "
            f"Available: {sorted(VIDEO_MODEL_BUILDERS.keys())}"
        )

    return VIDEO_MODEL_BUILDERS[key]