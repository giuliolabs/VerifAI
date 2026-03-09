"""
Xception CNN Baseline for Visual Deepfake Detection (Frame-Level)
==================================================================

This module defines the Xception-based visual baseline model used
for frame-level deepfake detection in VerifAI.

Xception is widely used in deepfake research due to:
- Strong performance on manipulation datasets (e.g. FaceForensics++)
- Depthwise separable convolutions (efficient + expressive)
- Proven effectiveness for detecting subtle facial artefacts

Implementation details:
- Uses timm implementation for stability and reproducibility
- Outputs a single logit for binary classification (real vs fake)
- Input resolution: 299x299 (standard for Xception)

Dependencies:
    pip install torch torchvision timm

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

import torch.nn as nn


def build_xception_binary(pretrained: bool = True) -> nn.Module:
    """
    Build Xception model with binary output head.

    Args:
        pretrained: whether to load ImageNet pretrained weights

    Returns:
        torch.nn.Module configured for binary classification
    """

    # timm provides a stable and well-tested implementation of the legacy Xception architecture.
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "Xception requires timm. Please install with: pip install timm"
        ) from e

    # num_classes=1 ensures the model outputs a single logit.
    # This is suitable for binary classification with BCEWithLogitsLoss.
    model = timm.create_model(
        "legacy_xception",
        pretrained=pretrained,
        num_classes=1  # single binary logit
    )

    return model