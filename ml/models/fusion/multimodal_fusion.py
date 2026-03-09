"""
Multimodal Audio-Visual Fusion Model (Late Fusion v1)
=====================================================

This module implements the first version of the multimodal
audio-visual fusion model used in VerifAI. It combines
independent visual and audio feature extractors and performs
late fusion at the embedding level.

Architecture
------------

Video branch:
- MobileNetV2 backbone (ImageNet pretrained)
- Processes 5 frames independently
- Averages frame embeddings

Audio branch:
- ResNet18 backbone adapted for 1-channel MFCC
- Processes MFCC tensor [B, 1, 40, T]

Fusion:
- Concatenate video + audio embeddings
- MLP classifier -> binary logit

Inputs
------
video: [B, 5, 3, 224, 224]
audio: [B, 1, 40, T]

Output
------
logits: [B]

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# PyTorch core modules
import torch
import torch.nn as nn

# torchvision provides pretrained CNN backbones
import torchvision.models as models


class MultimodalFusionModel(nn.Module):
    """
    Late-fusion multimodal deepfake detector.

    The design intentionally keeps audio and video encoders separate
    so each modality can learn meaningful features independently
    before being combined at the embedding level.
    """

    def __init__(
        self,
        video_embed_dim: int = 1280,   # MobileNetV2 output feature dimension
        audio_embed_dim: int = 512,    # ResNet18 output feature dimension
        fusion_hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()

        # -------------------------
        # Video encoder (MobileNetV2)
        # -------------------------

        # Load pretrained MobileNetV2 backbone.
        # ImageNet weights help transfer general visual feature learning.
        video_backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # Use convolutional feature extractor only (exclude classifier head)
        self.video_encoder = video_backbone.features

        # Adaptive pooling ensures fixed-size embedding regardless of input resolution
        self.video_pool = nn.AdaptiveAvgPool2d((1, 1))

        # -------------------------
        # Audio encoder (ResNet18 MFCC)
        # -------------------------

        # Load pretrained ResNet18 backbone
        audio_backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Adapt first convolution layer from RGB (3 channels) to MFCC (1 channel)
        # This allows the model to process spectrogram-like input.
        old_conv = audio_backbone.conv1
        audio_backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Initialize new 1-channel weights by averaging original RGB filters.
        # This preserves useful pretrained structure rather than random init.
        with torch.no_grad():
            audio_backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        # Remove final classification layer to obtain embedding only
        audio_backbone.fc = nn.Identity()

        self.audio_encoder = audio_backbone

        # -------------------------
        # Fusion head
        # -------------------------

        # Fusion is performed by concatenating both embeddings
        # followed by a small MLP for binary classification.
        self.fusion_head = nn.Sequential(
            nn.Linear(video_embed_dim + audio_embed_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

        # These attributes store modality embedding norms
        # (useful for monitoring modality dominance during training)
        self.last_video_norm_mean = None
        self.last_audio_norm_mean = None

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video: [B, 5, 3, 224, 224]
            audio: [B, 1, 40, T]

        Returns:
            logits: [B]
        """

        B, N, C, H, W = video.shape

        # -------------------------
        # Video branch
        # -------------------------

        # Merge batch and frame dimensions so each frame is processed independently
        video = video.view(B * N, C, H, W)

        # Extract spatial features
        v = self.video_encoder(video)

        # Global pooling to obtain frame-level embedding
        v = self.video_pool(v).flatten(1)   # [B*N, 1280]

        # Reshape back to [B, N, embed_dim] and average across frames
        # This produces a single video-level representation.
        v = v.view(B, N, -1).mean(dim=1)     # [B, 1280]

        # -------------------------
        # Audio branch
        # -------------------------

        # Process MFCC tensor to obtain audio embedding
        a = self.audio_encoder(audio)        # [B, 512]

        # -------------------------
        # Fusion logging
        # -------------------------

        # Track average L2 norm of each modality embedding.
        # This helps detect imbalance where one modality dominates fusion.
        with torch.no_grad():
            self.last_video_norm_mean = v.norm(dim=1).mean().item()
            self.last_audio_norm_mean = a.norm(dim=1).mean().item()

        # -------------------------
        # Fusion
        # -------------------------

        # Concatenate embeddings along feature dimension
        fused = torch.cat([v, a], dim=1)

        # Pass through fusion MLP to produce binary logit
        logits = self.fusion_head(fused).squeeze(1)

        return logits