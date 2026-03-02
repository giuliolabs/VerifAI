"""
Augmentations for Deepfake Frame Classification (FaceForensics++ etc.)

Goals:
- Improve generalization across compressions (C23/C40), lighting, and camera quality
- Reduce overfitting to dataset-specific artifacts
- Provide train / val transforms as a single source of truth

Dependencies:
    pip install torch torchvision pillow

Usage in train.py:
    from ml.augmentations import get_transforms
    train_tfms = get_transforms(train=True)
    val_tfms   = get_transforms(train=False)
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image, ImageFilter
import torch
from torchvision import transforms


# -----------------------------
# Custom PIL-level augmentations
# -----------------------------

class RandomJPEGCompression:
    """
    Re-encode image as JPEG with random quality to simulate compression artifacts.
    This is extremely relevant for FaceForensics++ (C23/C40) style distortions.
    """
    def __init__(self, quality_range: Tuple[int, int] = (35, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        qmin, qmax = self.quality_range
        quality = random.randint(qmin, qmax)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        out = Image.open(buffer).convert("RGB")
        return out


class RandomGaussianBlur:
    """
    Random blur to simulate camera focus/motion blur.
    """
    def __init__(self, radius_range: Tuple[float, float] = (0.1, 1.5), p: float = 0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        rmin, rmax = self.radius_range
        radius = random.uniform(rmin, rmax)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomAdditiveNoise:
    """
    Adds mild Gaussian noise in tensor space.
    """
    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.06), p: float = 0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        smin, smax = self.std_range
        std = random.uniform(smin, smax)
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0.0, 1.0)


@dataclass
class AugmentationConfig:
    image_size: int = 224

    # Geometry
    hflip_p: float = 0.5
    random_crop_scale: Tuple[float, float] = (0.85, 1.0)

    # Photometric
    color_jitter_p: float = 0.7
    brightness: float = 0.20
    contrast: float = 0.20
    saturation: float = 0.15
    hue: float = 0.03

    # Compression/blur/noise (deepfake-relevant)
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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool, cfg: Optional[AugmentationConfig] = None):
    """
    Returns torchvision transforms.
    - train=True: includes strong, realistic augmentations
    - train=False: deterministic resize + normalize
    """
    if cfg is None:
        cfg = AugmentationConfig()

    if not train:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # Train transforms (PIL -> Tensor -> Tensor aug -> Normalize)
    return transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomResizedCrop(
            size=cfg.image_size,
            scale=cfg.random_crop_scale,
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=cfg.hflip_p),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
                saturation=cfg.saturation,
                hue=cfg.hue
            )
        ], p=cfg.color_jitter_p),

        RandomJPEGCompression(quality_range=cfg.jpeg_quality_range, p=cfg.jpeg_p),
        RandomGaussianBlur(radius_range=cfg.blur_radius_range, p=cfg.blur_p),

        transforms.ToTensor(),

        RandomAdditiveNoise(std_range=cfg.noise_std_range, p=cfg.noise_p),

        transforms.RandomErasing(
            p=cfg.random_erasing_p,
            scale=cfg.erasing_scale,
            ratio=cfg.erasing_ratio,
            value="random"
        ),

        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
