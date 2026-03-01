"""
Model registry + factory (aligned with this repo structure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


__all__ = ["ModelSpec", "list_models", "get_model_spec", "create_model"]

Builder = Callable[..., Any]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str              # "video" | "audio" | "fusion"
    builder_path: str      # "module.path:callable"
    description: str = ""


_REGISTRY: Dict[str, ModelSpec] = {
    # ---- Video ----
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

    # ---- Audio ----
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

    # ---- Fusion ----
    "multimodal_fusion": ModelSpec(
        name="multimodal_fusion",
        task="fusion",
        builder_path="ml.models.fusion.multimodal_fusion:build_model",
        description="Audio-visual fusion model combining audio + video streams.",
    ),
}


def list_models(task: Optional[str] = None) -> Tuple[str, ...]:
    if task is None:
        return tuple(sorted(_REGISTRY.keys()))
    return tuple(sorted([n for n, s in _REGISTRY.items() if s.task == task]))


def get_model_spec(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(list_models())}")
    return _REGISTRY[name]


def _import_from_path(path: str) -> Builder:
    if ":" not in path:
        raise ValueError(f"Invalid builder_path '{path}'. Expected 'module:callable'.")
    module_path, attr = path.split(":", 1)
    module = __import__(module_path, fromlist=[attr])
    fn = getattr(module, attr, None)
    if fn is None:
        raise ImportError(f"Could not find '{attr}' inside '{module_path}'.")
    if not callable(fn):
        raise TypeError(f"Imported '{module_path}:{attr}' is not callable.")
    return fn


def create_model(name: str, **kwargs: Any) -> Any:
    spec = get_model_spec(name)
    builder = _import_from_path(spec.builder_path)
    return builder(**kwargs)
