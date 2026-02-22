"""
Raw Waveform Encoder for Deepfake Detection (Audio)
===================================================

Purpose
-------
Provides a simple, reliable baseline encoder that consumes raw audio waveforms
and produces:
- an embedding vector (for fusion or analysis), OR
- a binary logit (real vs fake)

Why this exists (VerifAI)
-------------------------
VerifAI currently supports MFCC-based audio detection (mfcc_cnn.py).
This module enables:
- Ablation: MFCC vs end-to-end waveform learning
- Future work: improved audio modeling without handcrafted features
- Fusion: swap MFCC input with learned audio embeddings

Input / Output
--------------
Input:
    x: torch.Tensor of shape [B, 1, T] or [B, T]
        - mono waveform
        - T is number of samples (can vary, but batching assumes fixed T)
Output:
    logits: [B, 1] if classification head enabled
    embedding: [B, D] if return_embedding=True

Dependencies:
    pip install torch

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class WavEncoderConfig:
    sample_rate: int = 16000
    embedding_dim: int = 256
    base_channels: int = 32
    dropout: float = 0.2


class WavEncoder1D(nn.Module):
    """
    Lightweight 1D CNN waveform encoder.

    Architecture (simple + stable):
    - Conv1D blocks with stride/downsampling
    - Global average pooling
    - MLP projection to embedding_dim

    Designed as a baseline for coursework/dissertation:
    easy to explain + easy to train.
    """

    def __init__(self, cfg: WavEncoderConfig):
        super().__init__()
        self.cfg = cfg

        c = cfg.base_channels

        # Feature extractor
        self.features = nn.Sequential(
            self._block(1, c,   k=9, s=2, p=4),     # /2
            self._block(c, c*2, k=9, s=2, p=4),     # /4
            self._block(c*2, c*4, k=9, s=2, p=4),   # /8
            self._block(c*4, c*8, k=9, s=2, p=4),   # /16
        )

        self.proj = nn.Sequential(
            nn.Linear(c * 8, cfg.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int, k: int, s: int, p: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            embedding: [B, embedding_dim]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B,1,T]
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,T] or [B,1,T], got {tuple(x.shape)}")

        feats = self.features(x)  # [B,C,T']
        pooled = feats.mean(dim=-1)  # global avg pool -> [B,C]
        emb = self.proj(pooled)      # [B,D]
        return emb


class WavBinaryClassifier(nn.Module):
    """
    Wraps WavEncoder1D with a binary classification head (logit output).
    """

    def __init__(self, cfg: Optional[WavEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg or WavEncoderConfig()
        self.encoder = WavEncoder1D(self.cfg)
        self.head = nn.Linear(self.cfg.embedding_dim, 1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [B,1]
            optionally also embedding: [B,D]
        """
        emb = self.encoder(x)
        logits = self.head(emb)
        if return_embedding:
            return logits, emb
        return logits


def build_wav_binary_classifier(
    sample_rate: int = 16000,
    embedding_dim: int = 256,
    base_channels: int = 32,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Convenience builder, similar style to build_mfcc_resnet18_binary().
    """
    cfg = WavEncoderConfig(
        sample_rate=sample_rate,
        embedding_dim=embedding_dim,
        base_channels=base_channels,
        dropout=dropout,
    )
    return WavBinaryClassifier(cfg)
