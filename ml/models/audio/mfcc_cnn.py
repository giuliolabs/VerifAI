"""
MFCC-based Audio CNN for Deepfake Detection.

Architecture:
- MFCC features treated as 2D inputs (1 x n_mfcc x time)
- ResNet-18 backbone adapted for single-channel input
- Binary classification (real vs fake)

Inspired by:
Khalid et al., 2021 – Audio-visual deepfake detection using MFCC features.

Dependencies:
    pip install torch torchvision

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import torch
import torch.nn as nn
import torchvision.models as models


def build_mfcc_resnet18_binary(pretrained: bool = True) -> nn.Module:
    """
    Build ResNet-18 adapted for MFCC input and binary classification.

    Returns:
        torch.nn.Module with output shape [B, 1]
    """
    model = models.resnet18(pretrained=pretrained)

    # Store original conv layer
    old_conv = model.conv1

    # Replace first conv layer: RGB (3) -> MFCC (1)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    # Initialize MFCC conv weights from pretrained RGB weights
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
            )

    # Replace classification head (binary output)
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
