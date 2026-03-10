"""
Train ViT Baseline on FaceForensics++ C23 Sampled Frames
========================================================

This script trains the Vision Transformer (ViT) baseline on frame-level
samples extracted from the FaceForensics++ C23 dataset. It is designed
to work with hashed/safe-id frame folders produced by the frame
extraction pipeline.

Folder structure expected:
data/interim/frames/FaceForensics++_C23/
  train/<safe_id>/frame_000.jpg ...
  train/_id_map.csv
  val/<safe_id>/frame_000.jpg ...
  val/_id_map.csv

Split CSV expected:
data/splits/faceforensics++_c23_train.csv with columns: video_path,label,video_id
data/splits/faceforensics++_c23_val.csv   with columns: video_path,label,video_id

Outputs:
experiments/results/ffpp_c23_vit_baseline/best_model.pt
experiments/logs/training_curves/vit_baseline.csv

This training setup joins:
- safe_id -> video_path using _id_map.csv
- video_path -> label using the split CSV

This keeps extracted folders anonymised while still preserving correct labels.

Dependencies:
    python -m pip install torch torchvision timm pandas pillow tqdm numpy

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path so internal modules can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import ViT model builder
from ml.models.video.vit import build_vit_binary

# Utility used to log epoch-level training curves
from scripts.curve_writer import append_curve_row


# -------------------------
# Configuration
# -------------------------

# Root folder containing extracted frames
FRAMES_ROOT = Path("data/interim/frames/FaceForensics++_C23")

# Folder containing train/validation split CSV files
SPLITS_DIR = Path("data/splits")

TRAIN_CSV = SPLITS_DIR / "faceforensics++_c23_train.csv"
VAL_CSV   = SPLITS_DIR / "faceforensics++_c23_val.csv"

# Output directory for best model checkpoint
OUT_DIR  = Path("experiments/results/ffpp_c23_vit_baseline")

# Name used for logging
RUN_NAME = "vit_baseline"

# Standard ViT input resolution
IMG_SIZE = 224

# Training configuration
MAX_EPOCHS = 15
PATIENCE = 3
MIN_DELTA = 1e-4

LR = 3e-5
BATCH_TRAIN = 16
BATCH_VAL = 32
NUM_WORKERS = 0
SEED = 42


def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.

    This helps keep the training procedure more stable and repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight_from_samples(samples: list[tuple[Path, int]]) -> float:
    """
    Compute positive class weight for BCEWithLogitsLoss.

    Standard formula:
        pos_weight = (#negative / #positive)

    This can help when the dataset is imbalanced.
    """
    n_pos = sum(1 for _, label in samples if int(label) == 1)
    n_neg = sum(1 for _, label in samples if int(label) == 0)

    if n_pos == 0:
        return 1.0

    return float(n_neg) / float(n_pos)


def make_transforms(train: bool) -> transforms.Compose:
    """
    Build image preprocessing pipeline.

    Training includes light augmentation,
    while validation uses deterministic preprocessing only.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def read_split_labels(split_csv: Path) -> dict[str, int]:
    """
    Read mapping from video_path -> label from split CSV.

    video_path is used as the stable join key when linking extracted
    safe_id folders back to their ground-truth labels.
    """
    mapping: dict[str, int] = {}

    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vp = str(row.get("video_path", "")).replace("\\", "/")
            if not vp:
                continue
            mapping[vp] = int(row["label"])

    return mapping


def read_id_map(id_map_path: Path) -> dict[str, str]:
    """
    Read <split>/_id_map.csv produced by extract_frames_from_csv.py.

    Returns mapping:
        safe_id -> normalized video_path

    This is needed because extracted frame folders are stored using
    anonymized safe_id names rather than original filenames.
    """
    safe_to_vpath: dict[str, str] = {}

    with id_map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            safe_id = str(row["safe_id"]).strip()
            vpath = str(row["video_path"]).replace("\\", "/").strip()
            if safe_id and vpath:
                safe_to_vpath[safe_id] = vpath

    return safe_to_vpath


class FFPPHashedFrameDataset(Dataset):
    """
    Dataset for loading individual extracted frames from hashed folders.

    Frames are read from:
      <frames_root>/<split>/<safe_id>/frame_*.jpg

    Labels are matched via:
      safe_id -> video_path (from _id_map.csv)
      video_path -> label (from split CSV)
    """

    def __init__(self, split: str, transform: transforms.Compose):
        self.split = split
        self.transform = transform

        split_dir = FRAMES_ROOT / split
        self.id_map_path = split_dir / "_id_map.csv"

        if not self.id_map_path.exists():
            raise FileNotFoundError(
                f"Missing id map: {self.id_map_path}. Re-run extract_frames_from_csv.py"
            )

        # Select the correct split CSV for the requested split
        split_csv = TRAIN_CSV if split == "train" else VAL_CSV

        # Build mappings required for the label join
        video_path_to_label = read_split_labels(split_csv)
        safe_to_vpath = read_id_map(self.id_map_path)

        self.samples: list[tuple[Path, int]] = []

        # Build final dataset samples as (frame_path, label)
        for safe_id, vpath in safe_to_vpath.items():
            if vpath not in video_path_to_label:
                continue

            label = int(video_path_to_label[vpath])
            vid_folder = split_dir / safe_id

            if not vid_folder.exists():
                continue

            for img_path in sorted(vid_folder.glob("frame_*.jpg")):
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No frames found for split={split}. "
                f"Check {split_dir} contains <safe_id>/frame_*.jpg and _id_map.csv matches split CSV."
            )

    def __len__(self) -> int:
        """Return total number of frame samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Load one frame and return:
        - image tensor
        - label tensor
        - image path string for traceability
        """
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y, str(img_path)


def main() -> None:
    """
    Main training pipeline for the ViT baseline.

    This function:
    - builds train/validation datasets
    - handles class imbalance using weighted sampling
    - applies a warmup stage for the classifier head
    - trains and validates the model
    - saves the best checkpoint
    - logs training curves
    """
    set_seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUT_DIR / "best_model.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Build datasets
    train_ds = FFPPHashedFrameDataset("train", transform=make_transforms(train=True))
    val_ds   = FFPPHashedFrameDataset("val",   transform=make_transforms(train=False))

    print("Train frames:", len(train_ds))
    print("Val frames:", len(val_ds))

    # ---- Imbalance handling ----
    # Compute class weight information from training samples.
    # This can be useful for weighted loss design, although this script
    # mainly uses weighted sampling to balance batches.
    pos_w = compute_pos_weight_from_samples(train_ds.samples)
    torch.tensor([pos_w], device=device, dtype=torch.float32)

    # Standard binary classification loss
    loss_fn = nn.BCEWithLogitsLoss()

    # ---- Weighted sampler to balance classes per batch ----
    labels = [label for _, label in train_ds.samples]
    class_counts = np.bincount(np.array(labels, dtype=np.int64), minlength=2)
    class_counts = np.maximum(class_counts, 1)

    # Inverse-frequency weighting gives higher sampling probability
    # to underrepresented classes.
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[int(l)] for l in labels]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_TRAIN,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_VAL,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Build pretrained ViT baseline
    model = build_vit_binary(pretrained=True, img_size=IMG_SIZE).to(device)

    # ---- Warmup stage: train classifier head only ----
    # Freezing the encoder at the beginning can make transfer learning
    # more stable, especially for large pretrained transformer models.
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        # Safety fallback in case model structure differs
        for param in model.parameters():
            param.requires_grad = True

    warmup_epochs = 3
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    curve_csv = f"experiments/logs/training_curves/{RUN_NAME}.csv"

    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        # ---- Unfreeze full model after warmup ----
        if epoch == warmup_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        # ---- Training phase ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} - train"):
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)

        # ---- Validation phase ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} - val"):
                images = images.to(device)
                labels = labels.float().to(device)

                logits = model(images).squeeze(1)
                loss = loss_fn(logits, labels)

                bs = images.size(0)
                val_loss_sum += float(loss.item()) * bs
                val_count += bs

        val_loss = val_loss_sum / max(val_count, 1)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ---- Checkpointing + early stopping ----
        improved = (best_val_loss - val_loss) > MIN_DELTA

        if improved:
            best_val_loss = val_loss
            bad_epochs = 0

            # Save best model checkpoint together with core configuration
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": float(best_val_loss),
                    "epoch": epoch,
                    "config": {
                        "model": "vit",
                        "img_size": IMG_SIZE,
                        "max_epochs": MAX_EPOCHS,
                        "patience": PATIENCE,
                        "min_delta": MIN_DELTA,
                        "lr": LR,
                        "seed": SEED,
                        "pos_weight": float(pos_w),
                        "sampler": "WeightedRandomSampler",
                        "warmup_epochs": warmup_epochs,
                    },
                },
                best_path,
            )
            print("Saved best ->", best_path)

        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

        # Log epoch metrics for later plotting/reporting
        append_curve_row(curve_csv, {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

    print("Done. Best val loss:", best_val_loss)
    print("Curves ->", curve_csv)


if __name__ == "__main__":
    main()