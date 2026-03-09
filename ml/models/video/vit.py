"""
Vision Transformer (ViT) for Frame-Based Deepfake Detection (Binary)
=====================================================================

This module implements a Vision Transformer (ViT) model for
frame-level deepfake detection within the VerifAI framework.

Key design decisions:
- Uses timm library models (e.g. vit_base_patch16_224)
- Supports feature-extractor mode for temporal modelling
- Supports binary classification via a single-logit head

The feature extractor mode is particularly important for:
- Temporal Transformer models
- Multimodal fusion experiments
- Embedding analysis

Dependencies:
    python -m pip install timm torch

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import torch
import torch.nn as nn

# timm provides flexible access to many ViT variants
try:
    import timm
except Exception as exc:
    raise ImportError(
        "Missing dependency: timm. Install with: python -m pip install timm"
    ) from exc


# ------------------------------------------------
# Feature Extractor Wrapper
# ------------------------------------------------

class ViTFeatureExtractor(nn.Module):
    """
    Wraps a timm ViT-like model in feature-extractor mode.

    Output:
        embeddings [B, D]

    This abstraction ensures compatibility across different
    timm ViT variants.
    """

    def __init__(self, timm_model: nn.Module):
        super().__init__()
        self.model = timm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings from the ViT backbone.
        """

        # Preferred approach: use forward_features if available
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            # Fallback if model already returns features directly
            feats = self.model(x)

        # Some timm models may return tuple or list
        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        # ViT models may return:
        # - [B, D] if already pooled
        # - [B, N, D] token sequence (CLS + patches)
        if feats.ndim == 3:
            # Use CLS token representation by default
            feats = feats[:, 0, :]

        # Ensure final shape is valid embedding format
        if feats.ndim != 2:
            raise RuntimeError(
                f"Expected [B,D] features after pooling, got shape: {tuple(feats.shape)}"
            )

        return feats


# ------------------------------------------------
# Binary Classification Wrapper
# ------------------------------------------------

class ViTBinaryClassifier(nn.Module):
    """
    ViT feature extractor + linear classification head.

    Produces a single logit per image: [B,1]
    """

    def __init__(self, encoder: nn.Module, embed_dim: int):
        super().__init__()

        self.encoder = encoder

        # Single logit output suitable for BCEWithLogitsLoss
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)      # [B, D]
        logits = self.head(feats)    # [B, 1]
        return logits


# ------------------------------------------------
# Builder: Feature Extractor
# ------------------------------------------------

def build_vit_feature_extractor(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    img_size: int = 224,
) -> tuple[nn.Module, int]:
    """
    Build a ViT model in feature-extractor mode.

    Returns:
        (feature_extractor, embed_dim)
    """

    # num_classes=0 removes classification head,
    # global_pool="avg" ensures pooled feature output
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        img_size=img_size,
        global_pool="avg",
    )

    # Most timm models expose embedding dimension via num_features
    embed_dim = int(getattr(backbone, "num_features", 0))

    if embed_dim <= 0:
        # Fallback: infer embedding dimension via dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feats = backbone(dummy)

            if isinstance(feats, (tuple, list)):
                feats = feats[0]

            embed_dim = int(feats.shape[-1])

    return ViTFeatureExtractor(backbone), embed_dim


# ------------------------------------------------
# Builder: Binary Classifier
# ------------------------------------------------

def build_vit_binary(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    img_size: int = 224,
) -> nn.Module:
    """
    Build ViT binary classifier.

    Output:
        logits [B, 1]
    """

    # First build encoder and retrieve embedding dimension
    encoder, dim = build_vit_feature_extractor(
        model_name=model_name,
        pretrained=pretrained,
        img_size=img_size,
    )

    # Attach binary classification head
    return ViTBinaryClassifier(encoder, dim)