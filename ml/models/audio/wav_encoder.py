"""
Raw Waveform Encoder for Deepfake Detection (Audio)
===================================================

This module implements a lightweight 1D convolutional neural network
that processes raw audio waveforms directly, without handcrafted features
such as MFCCs. It serves as an alternative audio modeling strategy
within the VerifAI framework.

The encoder can be used in two ways:
- As a feature extractor (embedding output)
- As a standalone binary classifier (real vs fake)

Why this exists (VerifAI)
-------------------------
While VerifAI primarily uses MFCC-based audio detection, this module enables:
- Ablation experiments (MFCC vs raw waveform learning)
- Exploration of end-to-end audio modeling
- Future multimodal fusion using learned waveform embeddings

Input / Output
--------------
Input:
    x: torch.Tensor of shape [B, 1, T] or [B, T]
        - mono waveform
        - T is number of samples

Output:
    logits: [B, 1] if classification head enabled
    embedding: [B, D] if return_embedding=True

Dependencies:
    pip install torch

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# dataclass is used for clean configuration management
from dataclasses import dataclass

# Optional allows configuration to be provided or defaulted
from typing import Optional, Tuple

# PyTorch core modules
import torch
import torch.nn as nn


@dataclass
class WavEncoderConfig:
    """
    Configuration container for the waveform encoder.

    This keeps hyperparameters grouped and easily adjustable.
    """
    sample_rate: int = 16000
    embedding_dim: int = 256
    base_channels: int = 32
    dropout: float = 0.2


class WavEncoder1D(nn.Module):
    """
    Lightweight 1D CNN waveform encoder.

    Architecture:
    - Several Conv1D blocks with stride-based downsampling
    - Global average pooling across time
    - MLP projection to fixed embedding dimension

    Designed as a simple and stable baseline model that is:
    - Easy to explain academically
    - Computationally efficient
    - Suitable for ablation experiments
    """

    def __init__(self, cfg: WavEncoderConfig):
        super().__init__()
        self.cfg = cfg

        c = cfg.base_channels

        # Feature extraction backbone.
        # Each block halves temporal resolution via stride=2,
        # progressively increasing channel depth.
        self.features = nn.Sequential(
            self._block(1, c,     k=9, s=2, p=4),      # Temporal /2
            self._block(c, c*2,   k=9, s=2, p=4),      # /4
            self._block(c*2, c*4, k=9, s=2, p=4),      # /8
            self._block(c*4, c*8, k=9, s=2, p=4),      # /16
        )

        # Projection layer maps pooled feature vector
        # to a fixed-size embedding representation.
        self.proj = nn.Sequential(
            nn.Linear(c * 8, cfg.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int, k: int, s: int, p: int) -> nn.Sequential:
        """
        Define a standard Conv1D block:
        Conv -> BatchNorm -> ReLU

        BatchNorm stabilizes training,
        ReLU introduces non-linearity.
        """
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            embedding: [B, embedding_dim]
        """

        # Accept both [B,T] and [B,1,T] formats for flexibility
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Convert to [B,1,T]

        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,T] or [B,1,T], got {tuple(x.shape)}")

        # Extract hierarchical temporal features
        feats = self.features(x)  # [B,C,T']

        # Global average pooling over time dimension
        # Produces fixed-size vector independent of waveform length
        pooled = feats.mean(dim=-1)  # [B,C]

        # Project to embedding space
        emb = self.proj(pooled)      # [B,D]

        return emb


class WavBinaryClassifier(nn.Module):
    """
    Wraps WavEncoder1D with a binary classification head.

    This allows the same encoder to be reused:
    - Either as an embedding extractor
    - Or as a full classifier
    """

    def __init__(self, cfg: Optional[WavEncoderConfig] = None):
        super().__init__()

        # Use default config if none provided
        self.cfg = cfg or WavEncoderConfig()

        # Base encoder
        self.encoder = WavEncoder1D(self.cfg)

        # Final linear layer produces a single logit
        # suitable for binary classification
        self.head = nn.Linear(self.cfg.embedding_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [B,1]
            optionally embedding: [B,D]
        """

        # Extract embedding
        emb = self.encoder(x)

        # Compute classification logit
        logits = self.head(emb)

        # Optionally return both outputs for fusion experiments
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
    Convenience builder function, similar in style to
    build_mfcc_resnet18_binary().

    This keeps model creation consistent across the VerifAI codebase.
    """

    cfg = WavEncoderConfig(
        sample_rate=sample_rate,
        embedding_dim=embedding_dim,
        base_channels=base_channels,
        dropout=dropout,
    )

    return WavBinaryClassifier(cfg)