"""
Temporal Transformer Video Classifier (binary) for deepfake detection.

This model turns a sequence of frames into a single video-level prediction.

Input:
    frames: [B, T, 3, H, W]
Output:
    logits: [B, 1]

Supports backbones that produce embeddings [B, D], such as:
- ViT feature extractor from vit.py
- MobileNetV2 feature extractor

Dependencies:
    torch

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Temporal transformer encoder over frame embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # [B,T,D]
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        h = self.encoder(x)
        h = self.norm(h)
        return h


class VideoTemporalClassifier(nn.Module):
    """
    frames [B,T,3,H,W] -> encoder -> embeddings [B,T,D] -> temporal transformer -> logit [B,1]
    """

    def __init__(
        self,
        frame_encoder: nn.Module,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        pool: str = "mean",  # "mean" or "cls" (cls means first token)
    ):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.temporal = TemporalTransformer(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.head = nn.Linear(embed_dim, 1)
        self.pool = pool

        if freeze_encoder:
            for p in self.frame_encoder.parameters():
                p.requires_grad = False

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B,T,3,H,W]
        returns: [B,T,D]
        """
        if frames.ndim != 5:
            raise ValueError(f"Expected frames [B,T,3,H,W], got {tuple(frames.shape)}")

        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)           # [B*T,3,H,W]
        feats = self.frame_encoder(flat)             # [B*T,D]

        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        if feats.ndim != 2:
            raise RuntimeError(f"Frame encoder must return [N,D]. Got: {tuple(feats.shape)}")

        D = feats.shape[1]
        feats = feats.view(B, T, D)                  # [B,T,D]
        return feats

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        emb = self._encode_frames(frames)            # [B,T,D]
        h = self.temporal(emb)                       # [B,T,D]

        if self.pool == "cls":
            pooled = h[:, 0, :]                      # [B,D]
        else:
            pooled = h.mean(dim=1)                   # [B,D]

        logits = self.head(pooled)                   # [B,1]
        return logits


# -------------------------
# Backbone helpers (usable NOW)
# -------------------------

def build_mobilenetv2_feature_extractor(pretrained: bool = True) -> tuple[nn.Module, int]:
    """
    MobileNetV2 -> feature extractor returning [B,D] (global pooled).
    """
    from torchvision import models

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
    features = model.features  # conv stack

    class MobileNetV2Encoder(nn.Module):
        def __init__(self, feats: nn.Module):
            super().__init__()
            self.feats = feats
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.feats(x)               # [B,C,h,w]
            h = self.pool(h).flatten(1)     # [B,C]
            return h

    # MobileNetV2 last channel count is 1280
    return MobileNetV2Encoder(features), 1280


def build_vit_feature_extractor(pretrained: bool = True, img_size: int = 224):
    """
    Delegates to your vit.py implementation.
    Returns (encoder, embed_dim).
    """
    from .vit import build_vit_feature_extractor as _build
    return _build(pretrained=pretrained, img_size=img_size)


def build_temporal_transformer_model(
    backbone: str = "mobilenetv2",   # "mobilenetv2" or "vit"
    pretrained: bool = True,
    img_size: int = 224,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
) -> nn.Module:
    """
    One-line builder for a complete video temporal transformer classifier.
    """
    name = backbone.strip().lower()

    if name == "mobilenetv2":
        enc, dim = build_mobilenetv2_feature_extractor(pretrained=pretrained)
    elif name == "vit":
        enc, dim = build_vit_feature_extractor(pretrained=pretrained, img_size=img_size)
    else:
        raise KeyError("Unknown backbone. Use: mobilenetv2 | vit")

    return VideoTemporalClassifier(
        frame_encoder=enc,
        embed_dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        pool="mean",
    )
