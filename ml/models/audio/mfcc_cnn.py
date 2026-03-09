"""
MFCC-based Audio CNN for Deepfake Detection
==========================================

This module defines the audio classification model used in VerifAI for
deepfake detection from MFCC representations. Instead of raw waveforms,
the model uses MFCC features as a compact time-frequency input, which
makes the problem suitable for a convolutional neural network.

Architecture:
- MFCC features treated as 2D inputs (1 x n_mfcc x time)
- ResNet-18 backbone adapted for single-channel input
- Binary classification (real vs fake)

Inspired by:
Khalid et al., 2021 – Audio-visual deepfake detection using MFCC features.

Dependencies:
    pip install torch torchvision

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# PyTorch core modules for defining and modifying neural networks
import torch
import torch.nn as nn

# torchvision provides the ResNet-18 backbone used as the base architecture
import torchvision.models as models


def build_mfcc_resnet18_binary(pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet-18 model adapted for MFCC input and binary classification.

    Returns:
        torch.nn.Module with output shape [B, 1]
    """

    # Load the standard ResNet-18 architecture.
    # Using pretrained ImageNet weights can help transfer useful low-level
    # feature patterns, even though the original model was designed for images.
    model = models.resnet18(pretrained=pretrained)

    # Save the original first convolution layer before replacing it.
    # This is useful because its weights can be reused to initialize the
    # new single-channel input layer more sensibly than random initialization.
    old_conv = model.conv1

    # Replace the first convolution layer so the network accepts
    # MFCC input with 1 channel instead of RGB input with 3 channels.
    # The other convolution settings are kept the same to preserve
    # the original backbone structure.
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    # If pretrained weights are being used, initialize the new
    # single-channel convolution by averaging the original RGB filters.
    # This keeps some useful learned structure from ImageNet while
    # adapting it to MFCC input.
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
            )

    # Replace the original classification head with a single-output layer.
    # A single logit is suitable for binary classification (real vs fake),
    # where sigmoid can later convert the output into a probability.
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model