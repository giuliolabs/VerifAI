"""
Video model registry for VerifAI.

Provides clean imports and a builder registry:
    from ml.models.video import get_video_model_builder

Models:
- xception
- mobilenetv2
- vit
- temporal (Temporal Transformer)

Author: Giulio Dajani
Project: VerifAI
"""

from __future__ import annotations

from typing import Callable, Dict

from .xception import build_xception_binary
from .mobilenet_baseline import build_mobilenet_v2_binary
from .vit import build_vit_binary
from .temporal_transformer import build_temporal_transformer_model


VIDEO_MODEL_BUILDERS: Dict[str, Callable[..., object]] = {
    "xception": build_xception_binary,
    "mobilenetv2": build_mobilenet_v2_binary,
    "vit": build_vit_binary,
    "temporal": build_temporal_transformer_model,
}


def get_video_model_builder(name: str):
    key = name.strip().lower()
    if key not in VIDEO_MODEL_BUILDERS:
        raise KeyError(
            f"Unknown video model: {name}. "
            f"Available: {sorted(VIDEO_MODEL_BUILDERS.keys())}"
        )
    return VIDEO_MODEL_BUILDERS[key]
