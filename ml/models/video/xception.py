"""
Xception CNN baseline for visual deepfake detection (frame-level).

Dependencies:
    pip install torch torchvision timm

Notes:
- Uses timm implementation for stability and reproducibility.
- Outputs a single logit for binary classification (real vs fake).
- Input resolution: 299x299 (standard for Xception).

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import torch.nn as nn


def build_xception_binary(pretrained: bool = True) -> nn.Module:
    """
    Build Xception model with binary output head.

    Returns:
        torch.nn.Module
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "Xception requires timm. Install with: pip install timm"
        ) from e

    model = timm.create_model(
        "legacy_xception",
        pretrained=pretrained,
        num_classes=1  # binary logit
    )
    return model
