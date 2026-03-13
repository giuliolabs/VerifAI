"""
Augmentations for Deepfake Frame Classification (FaceForensics++ etc.)
======================================================================

This module defines the image augmentation pipeline used for frame-based
deepfake detection in VerifAI. The aim is to improve generalization by
making training data more varied and closer to real-world conditions.

Goals:
- Improve generalization across compressions (C23/C40), lighting, and camera quality
- Reduce overfitting to dataset-specific artifacts
- Provide train / val transforms as a single source of truth

The augmentations are designed to be especially relevant for deepfake
datasets, where compression artifacts, blur, and low-quality frames
are common and can otherwise bias the model.

Dependencies:
    pip install torch torchvision pillow

Usage in train.py:
    from ml.augmentations import get_transforms
    train_tfms = get_transforms(train=True)
    val_tfms   = get_transforms(train=False)

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# io is used to temporarily save images in memory
# for JPEG recompression augmentation
import io

# random is used to apply stochastic augmentation behavior
import random

# dataclass stores augmentation settings clearly in one place
from dataclasses import dataclass

# Optional and Tuple improve readability of typed configuration fields
from typing import Optional, Tuple

# PIL is used for image-level augmentations before tensor conversion
from PIL import Image, ImageFilter

# torch is needed for tensor-space noise augmentation
import torch

# torchvision provides the standard augmentation pipeline tools
from torchvision import transforms


# -----------------------------
# Custom PIL-level augmentations
# -----------------------------

class RandomJPEGCompression:
    """
    Re-encode image as JPEG with random quality to simulate compression artifacts.

    This is particularly relevant for FaceForensics++ because fake videos are
    often distributed in compressed form, and compression can strongly affect
    visual artifacts.
    """

    def __init__(self, quality_range: Tuple[int, int] = (35, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        # Skip augmentation with probability (1 - p)
        if random.random() > self.p:
            return img

        qmin, qmax = self.quality_range
        quality = random.randint(qmin, qmax)

        # Re-save image in memory as JPEG with chosen quality
        # then reload it to simulate realistic compression loss
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        out = Image.open(buffer).convert("RGB")
        return out


class RandomGaussianBlur:
    """
    Apply random Gaussian blur to simulate mild focus or motion blur.
    """

    def __init__(self, radius_range: Tuple[float, float] = (0.1, 1.5), p: float = 0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        # Only blur some images, not all
        if random.random() > self.p:
            return img

        rmin, rmax = self.radius_range
        radius = random.uniform(rmin, rmax)

        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomAdditiveNoise:
    """
    Add mild Gaussian noise after tensor conversion.

    Noise is applied in tensor space because it is easier to control numerically.
    """

    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.06), p: float = 0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Skip augmentation with probability (1 - p)
        if random.random() > self.p:
            return x

        smin, smax = self.std_range
        std = random.uniform(smin, smax)

        # Add random Gaussian noise and clamp
        # so values stay inside valid image range [0,1]
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0.0, 1.0)


@dataclass
class AugmentationConfig:
    """
    Central configuration object for all augmentation settings.

    Keeping these values together makes it easier to:
    - adjust augmentation strength
    - reproduce experiments
    - document preprocessing clearly
    """
    image_size: int = 224

    # Geometry-related transforms
    hflip_p: float = 0.5
    random_crop_scale: Tuple[float, float] = (0.85, 1.0)

    # Photometric transforms
    color_jitter_p: float = 0.7
    brightness: float = 0.20
    contrast: float = 0.20
    saturation: float = 0.15
    hue: float = 0.03

    # Compression / blur / noise
    # These are especially relevant for deepfake robustness
    jpeg_p: float = 0.6
    jpeg_quality_range: Tuple[int, int] = (35, 95)

    blur_p: float = 0.25
    blur_radius_range: Tuple[float, float] = (0.1, 1.5)

    noise_p: float = 0.25
    noise_std_range: Tuple[float, float] = (0.0, 0.06)

    # Regularization
    random_erasing_p: float = 0.25
    erasing_scale: Tuple[float, float] = (0.02, 0.12)
    erasing_ratio: Tuple[float, float] = (0.3, 3.3)


# Standard ImageNet normalization values
# Used because most pretrained backbones were originally trained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool, cfg: Optional[AugmentationConfig] = None):
    """
    Build torchvision transforms for training or evaluation.

    Args:
        train:
            if True, return stronger augmentation pipeline
            if False, return deterministic resize + normalization only

        cfg:
            optional augmentation configuration object

    Returns:
        torchvision.transforms.Compose
    """
    if cfg is None:
        cfg = AugmentationConfig()

    # Validation / test pipeline:
    # deterministic and simple so evaluation stays consistent
    if not train:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # Training pipeline:
    # combines geometric, photometric, compression, blur, noise,
    # and regularization transforms
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),

        # RandomResizedCrop encourages robustness to framing/scale changes
        transforms.RandomResizedCrop(
            size=cfg.image_size,
            scale=cfg.random_crop_scale,
            ratio=(0.9, 1.1)
        ),

        # Horizontal flipping is useful because facial orientation
        # should not change the real/fake label
        transforms.RandomHorizontalFlip(p=cfg.hflip_p),

        # Randomly apply color jitter rather than always applying it,
        # to avoid making the data unrealistically distorted
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
                saturation=cfg.saturation,
                hue=cfg.hue
            )
        ], p=cfg.color_jitter_p),

        # Deepfake-relevant low-level artefact simulation
        RandomJPEGCompression(quality_range=cfg.jpeg_quality_range, p=cfg.jpeg_p),
        RandomGaussianBlur(radius_range=cfg.blur_radius_range, p=cfg.blur_p),

        # Convert to tensor before tensor-space transforms
        transforms.ToTensor(),

        # Add small Gaussian noise
        RandomAdditiveNoise(std_range=cfg.noise_std_range, p=cfg.noise_p),

        # Random erasing acts as extra regularization and reduces reliance
        # on small local artifacts only
        transforms.RandomErasing(
            p=cfg.random_erasing_p,
            scale=cfg.erasing_scale,
            ratio=cfg.erasing_ratio,
            value="random"
        ),

        # Final normalization for pretrained backbones
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])