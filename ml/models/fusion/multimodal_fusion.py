"""
Multimodal Audio-Visual Fusion Model (Late Fusion v1)

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

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        video_embed_dim: int = 1280,   # MobileNetV2 feature dim
        audio_embed_dim: int = 512,    # ResNet18 feature dim
        fusion_hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()

        # -------------------------
        # Video encoder (MobileNetV2)
        # -------------------------
        video_backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.video_encoder = video_backbone.features
        self.video_pool = nn.AdaptiveAvgPool2d((1, 1))

        # -------------------------
        # Audio encoder (ResNet18 MFCC)
        # -------------------------
        audio_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapt first conv to 1-channel MFCC
        old_conv = audio_backbone.conv1
        audio_backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            audio_backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        audio_backbone.fc = nn.Identity()
        self.audio_encoder = audio_backbone

        # -------------------------
        # Fusion head
        # -------------------------
        self.fusion_head = nn.Sequential(
            nn.Linear(video_embed_dim + audio_embed_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

        # Exposed for fusion logging
        self.last_video_norm_mean = None
        self.last_audio_norm_mean = None

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        video: [B, 5, 3, 224, 224]
        audio: [B, 1, 40, T]
        """

        B, N, C, H, W = video.shape

        # -------------------------
        # Video branch
        # -------------------------
        video = video.view(B * N, C, H, W)
        v = self.video_encoder(video)
        v = self.video_pool(v).flatten(1)   # [B*N, 1280]
        v = v.view(B, N, -1).mean(dim=1)     # average over frames -> [B, 1280]

        # -------------------------
        # Audio branch
        # -------------------------
        a = self.audio_encoder(audio)        # [B, 512]

        # -------------------------
        # Fusion logging (optional)
        # -------------------------
        with torch.no_grad():
            self.last_video_norm_mean = v.norm(dim=1).mean().item()
            self.last_audio_norm_mean = a.norm(dim=1).mean().item()

        # -------------------------
        # Fusion
        # -------------------------
        fused = torch.cat([v, a], dim=1)
        logits = self.fusion_head(fused).squeeze(1)

        return logits
