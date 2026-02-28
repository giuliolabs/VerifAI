"""
Vision Transformer (ViT) for frame-based deepfake detection (binary).

- Uses timm models (e.g. vit_base_patch16_224).
- Returns a single logit per image: [B, 1]
- Also exposes a feature extractor mode to get embeddings for temporal models.

Dependencies:
    python -m pip install timm torch

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except Exception as exc:
    raise ImportError("Missing dependency: timm. Install with: python -m pip install timm") from exc


class ViTFeatureExtractor(nn.Module):
    """
    Wraps a timm ViT-like model in feature-extractor mode.
    Output: [B, D]
    """

    def __init__(self, timm_model: nn.Module):
        super().__init__()
        self.model = timm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm convention: forward_features returns [B, D] for ViT-like models
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            # fallback: forward() may already return features if num_classes=0
            feats = self.model(x)

        # Some models may return tuple/list
        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        # timm ViT may return:
        # - [B, D] already pooled, or
        # - [B, N, D] tokens (CLS + patches)
        if feats.ndim == 3:
            # Use CLS token by default
            feats = feats[:, 0, :]  # [B, D]

        if feats.ndim != 2:
            raise RuntimeError(f"Expected [B,D] features after pooling, got shape: {tuple(feats.shape)}")

        return feats


class ViTBinaryClassifier(nn.Module):
    """
    ViT feature extractor + linear head -> binary logit [B,1]
    """

    def __init__(self, encoder: nn.Module, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)      # [B, D]
        logits = self.head(feats)    # [B, 1]
        return logits


def build_vit_feature_extractor(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    img_size: int = 224,
) -> tuple[nn.Module, int]:
    """
    Returns:
        (feature_extractor, embed_dim)
    """
    # num_classes=0 makes timm return features [no classification head]
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        img_size=img_size,
        global_pool="avg",
    )

    # embed dim usually accessible as num_features in timm
    embed_dim = int(getattr(backbone, "num_features", 0))
    if embed_dim <= 0:
        # fallback: try a dummy forward to infer dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feats = backbone(dummy)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            embed_dim = int(feats.shape[-1])

    return ViTFeatureExtractor(backbone), embed_dim


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
    encoder, dim = build_vit_feature_extractor(
        model_name=model_name,
        pretrained=pretrained,
        img_size=img_size,
    )
    return ViTBinaryClassifier(encoder, dim)
