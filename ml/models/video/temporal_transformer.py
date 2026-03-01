"""
Temporal Transformer (T=5 frames) for deepfake detection.

Pipeline:
  frames [B,T,3,H,W]
    -> frame encoder (ViT or MobileNetV2) produces embeddings [B,T,D]
    -> TransformerEncoder over time
    -> mean pool over time
    -> linear head -> logit [B,1]

Dependencies:
    pip install torch torchvision timm

Author: Giulio Dajani
Project: VerifAI
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameEncoderViT(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        import timm

        # num_classes=0 returns features instead of logits for many timm models
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.out_dim = getattr(self.backbone, "num_features", 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        feats = self.backbone(x)

        # timm may return [B,D] or [B,N,D]
        if feats.ndim == 3:
            feats = feats[:, 0, :]  # CLS token
        return feats  # [B,D]


class FrameEncoderMobileNetV2(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v2(weights=weights)

        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        h = self.features(x)
        h = self.pool(h).flatten(1)  # [B,1280]
        return h


class TemporalTransformer(nn.Module):
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

        if backbone == "vit":
            self.encoder = FrameEncoderViT(model_name=vit_name, pretrained=pretrained)
        elif backbone == "mobilenetv2":
            self.encoder = FrameEncoderMobileNetV2(pretrained=pretrained)
        else:
            raise ValueError("backbone must be 'vit' or 'mobilenetv2'")

        d_model = self.encoder.out_dim

        # Normalize embeddings before temporal encoder (helps stability)
        self.feat_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # IMPORTANT: input is [B,T,D]
            activation="gelu",
            norm_first=False,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B,T,3,H,W]
        returns: logits [B,1]
        """
        b, t, c, h, w = frames.shape
        x = frames.view(b * t, c, h, w)               # [B*T,3,H,W]
        feats = self.encoder(x)                       # [B*T,D]
        feats = feats.view(b, t, -1)                  # [B,T,D]
        feats = self.feat_norm(feats)
        feats = self.drop(feats)

        z = self.temporal(feats)                      # [B,T,D]
        z = z.mean(dim=1)                             # [B,D]
        logits = self.head(z)                         # [B,1]
        return logits


def build_temporal_transformer_model(
    backbone: str = "vit",
    pretrained: bool = True,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
) -> nn.Module:
    return TemporalTransformer(
        backbone=backbone,
        pretrained=pretrained,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    )
