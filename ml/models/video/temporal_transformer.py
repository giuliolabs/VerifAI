"""
Temporal Transformer (T=5 frames) for Deepfake Detection
=========================================================

This module implements a temporal modelling approach for deepfake
detection. Instead of averaging frame predictions, it explicitly
models temporal relationships between frames using a Transformer encoder.

Pipeline:
  frames [B,T,3,H,W]
    -> frame encoder (ViT or MobileNetV2) produces embeddings [B,T,D]
    -> TransformerEncoder over time
    -> mean pool over time
    -> linear head -> logit [B,1]

This allows the model to capture temporal inconsistencies across frames,
which are often present in manipulated videos.

Dependencies:
    pip install torch torchvision timm

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ------------------------------------------------
# Frame Encoder (Vision Transformer)
# ------------------------------------------------

class FrameEncoderViT(nn.Module):
    """
    Frame-level encoder using a Vision Transformer backbone.

    Extracts a single embedding per frame, typically using
    the CLS token representation.
    """

    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        import timm

        # num_classes=0 ensures timm returns feature embeddings
        # instead of classification logits.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        # Store output embedding dimension
        self.out_dim = getattr(self.backbone, "num_features", 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        feats = self.backbone(x)

        # Some ViT models return token sequence [B,N,D].
        # In that case, use the CLS token (index 0).
        if feats.ndim == 3:
            feats = feats[:, 0, :]

        return feats  # [B,D]


# ------------------------------------------------
# Frame Encoder (MobileNetV2)
# ------------------------------------------------

class FrameEncoderMobileNetV2(nn.Module):
    """
    Frame-level encoder using MobileNetV2 backbone.

    Provides a lightweight alternative to ViT.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v2(weights=weights)

        # Use convolutional features only
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # MobileNetV2 final embedding size
        self.out_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        h = self.features(x)
        h = self.pool(h).flatten(1)  # [B,1280]
        return h


# ------------------------------------------------
# Temporal Transformer Model
# ------------------------------------------------

class TemporalTransformer(nn.Module):
    """
    Temporal deepfake detector using frame embeddings + Transformer.

    The idea is:
    - Extract spatial features per frame
    - Model temporal relationships across frames
    - Aggregate temporal information for final classification
    """

    def __init__(
        self,
        backbone: str = "vit",
        pretrained: bool = True,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        vit_name: str = "vit_base_patch16_224",
    ):
        super().__init__()

        # Select frame encoder backbone
        if backbone == "vit":
            self.encoder = FrameEncoderViT(
                model_name=vit_name,
                pretrained=pretrained
            )
        elif backbone == "mobilenetv2":
            self.encoder = FrameEncoderMobileNetV2(
                pretrained=pretrained
            )
        else:
            raise ValueError("backbone must be 'vit' or 'mobilenetv2'")

        d_model = self.encoder.out_dim

        # Normalize embeddings before feeding into Transformer.
        # This improves numerical stability.
        self.feat_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # Define one Transformer encoder layer
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # Input shape is [B,T,D]
            activation="gelu",
            norm_first=False,
        )

        # Stack multiple temporal layers
        self.temporal = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

        # Final classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        # Optionally freeze spatial encoder to train only temporal layers
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B,T,3,H,W]
        returns: logits [B,1]
        """

        b, t, c, h, w = frames.shape

        # Merge batch and time dimensions to encode frames independently
        x = frames.view(b * t, c, h, w)        # [B*T,3,H,W]

        # Extract per-frame embeddings
        feats = self.encoder(x)                # [B*T,D]

        # Restore temporal structure
        feats = feats.view(b, t, -1)           # [B,T,D]

        # Normalize and apply dropout before Transformer
        feats = self.feat_norm(feats)
        feats = self.drop(feats)

        # Temporal modelling across frames
        z = self.temporal(feats)               # [B,T,D]

        # Aggregate temporal outputs (mean pooling)
        z = z.mean(dim=1)                      # [B,D]

        # Final binary logit
        logits = self.head(z)                  # [B,1]

        return logits


# ------------------------------------------------
# Builder Function
# ------------------------------------------------

def build_temporal_transformer_model(
    backbone: str = "vit",
    pretrained: bool = True,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
) -> nn.Module:
    """
    Convenience builder for consistency with the model registry.
    """
    return TemporalTransformer(
        backbone=backbone,
        pretrained=pretrained,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    )