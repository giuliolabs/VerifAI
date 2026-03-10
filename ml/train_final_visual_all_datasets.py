"""
Train Final Visual Backbone on All Available Frame Datasets
===========================================================

This script trains the final visual Xception backbone used in VerifAI
by combining frame-level samples from all available extracted frame
datasets. The goal is to improve generalisation by exposing the visual
model to a wider range of manipulations and recording conditions.

Datasets used:
- FaceForensics++_C23
- Celeb-DF-v2
- DeeperForensics
- FakeAVCeleb_v1.2

Strategy:
- train one Xception visual model across all frame datasets
- use hashed frame folders + split CSV + _id_map.csv
- cap frames per video for practical CPU training

Run:
    python -m ml.train_final_visual_all_datasets

Output:
    experiments/results/final_visual_all_datasets_xception/best_model.pt

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to the Python path so internal modules can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Standard libraries
import csv
import random
from dataclasses import dataclass

# Numerical computing
import numpy as np

# PyTorch core modules
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Image preprocessing tools
from torchvision import transforms
from PIL import Image

# Progress bars for training/validation
from tqdm import tqdm

# Project model builder
from ml.models.video.xception import build_xception_binary

# Utility for logging epoch-level training curves
from scripts.curve_writer import append_curve_row


@dataclass
class DatasetConfig:
    """
    Simple configuration object describing one dataset source.

    This keeps dataset paths grouped together so the training script
    can iterate through multiple datasets in a clean and scalable way.
    """
    name: str
    frames_root: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path


# List of all datasets included in final visual training.
# Each one contributes frames from the same train/val/test structure.
DATASETS = [
    DatasetConfig(
        name="FaceForensics++ C23",
        frames_root=Path("data/interim/frames/FaceForensics++_C23"),
        train_csv=Path("data/splits/faceforensics++_c23_train.csv"),
        val_csv=Path("data/splits/faceforensics++_c23_val.csv"),
        test_csv=Path("data/splits/faceforensics++_c23_test.csv"),
    ),
    DatasetConfig(
        name="Celeb-DF v2",
        frames_root=Path("data/interim/frames/Celeb-DF-v2"),
        train_csv=Path("data/splits/celebdfv2_train.csv"),
        val_csv=Path("data/splits/celebdfv2_val.csv"),
        test_csv=Path("data/splits/celebdfv2_test.csv"),
    ),
    DatasetConfig(
        name="DeeperForensics",
        frames_root=Path("data/interim/frames/DeeperForensics"),
        train_csv=Path("data/splits/deeperforensics_train.csv"),
        val_csv=Path("data/splits/deeperforensics_val.csv"),
        test_csv=Path("data/splits/deeperforensics_test.csv"),
    ),
    DatasetConfig(
        name="FakeAVCeleb",
        frames_root=Path("data/interim/frames/FakeAVCeleb_v1.2"),
        train_csv=Path("data/splits/fakeavceleb_train.csv"),
        val_csv=Path("data/splits/fakeavceleb_val.csv"),
        test_csv=Path("data/splits/fakeavceleb_test.csv"),
    ),
]

# Output directory for final checkpoint
OUT_DIR = Path("experiments/results/final_visual_all_datasets_xception")

# Run name used for curve logging
RUN_NAME = "final_visual_all_datasets_xception"

# Xception standard input resolution
IMG_SIZE = 299

# Training configuration
MAX_EPOCHS = 15
PATIENCE = 3
MIN_DELTA = 1e-4
LR = 1e-4
BATCH_TRAIN = 16
BATCH_VAL = 32
NUM_WORKERS = 0
SEED = 42

# Limit the number of frames taken from each video folder.
# This keeps training more practical while still sampling temporal variety.
MAX_FRAMES_PER_VIDEO = 5


def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(train: bool):
    """
    Build preprocessing pipeline for input frames.

    Training uses light augmentation to improve generalization,
    while validation uses deterministic transforms only.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
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
    Read mapping from video_path -> label from a split CSV.

    video_path is used as the stable join key between the dataset split
    file and the anonymized extracted frame folders.
    """
    mapping: dict[str, int] = {}

    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vp = str(row.get("video_path", "")).replace("\\", "/").strip()
            if not vp:
                continue
            mapping[vp] = int(row["label"])

    return mapping


def read_id_map(id_map_path: Path) -> dict[str, str]:
    """
    Read mapping from safe_id -> video_path from _id_map.csv.

    This allows hashed frame folders to be matched back to their
    correct labels through the original video_path.
    """
    mapping: dict[str, str] = {}

    with id_map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            safe_id = str(row["safe_id"]).strip()
            vpath = str(row["video_path"]).replace("\\", "/").strip()
            if safe_id and vpath:
                mapping[safe_id] = vpath

    return mapping


class MultiDatasetFrameDataset(Dataset):
    """
    Combined frame-level dataset built from multiple deepfake datasets.

    Each sample is an individual frame paired with a binary label.
    The dataset is created by iterating through all configured sources
    and joining:
    - safe_id -> video_path via _id_map.csv
    - video_path -> label via split CSV
    """

    def __init__(self, split: str, transform, max_frames_per_video: int = 5):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        # Iterate through every configured dataset and collect usable samples
        for cfg in DATASETS:
            split_csv = {
                "train": cfg.train_csv,
                "val": cfg.val_csv,
                "test": cfg.test_csv,
            }[split]

            split_dir = cfg.frames_root / split
            id_map_path = split_dir / "_id_map.csv"

            # Skip datasets that are incomplete rather than crashing immediately.
            # This makes the script more robust across different environments.
            if not split_dir.exists() or not id_map_path.exists() or not split_csv.exists():
                print(f"[WARN] Skipping {cfg.name} ({split}) due to missing paths.")
                continue

            label_map = read_split_labels(split_csv)
            safe_to_vpath = read_id_map(id_map_path)

            added = 0

            for safe_id, vpath in safe_to_vpath.items():
                if vpath not in label_map:
                    continue

                label = int(label_map[vpath])
                video_dir = split_dir / safe_id
                if not video_dir.exists():
                    continue

                # Limit frames per video so large datasets do not dominate training
                # and so CPU or GPU training remains practical.
                frame_paths = sorted(video_dir.glob("frame_*.jpg"))[:max_frames_per_video]

                for frame_path in frame_paths:
                    self.samples.append((frame_path, label))
                    added += 1

            print(f"{cfg.name} [{split}] -> {added} frames")

        if len(self.samples) == 0:
            raise FileNotFoundError("No frames found across datasets.")

    def __len__(self):
        """Return total number of collected frame samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load one frame sample and return:
        - transformed image tensor
        - label tensor
        - frame path string for traceability
        """
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y, str(img_path)


def main():
    """
    Main training pipeline for the final visual backbone.

    This function:
    - builds combined train/validation datasets across all sources
    - balances the training batches with weighted sampling
    - trains the Xception visual backbone
    - applies early stopping based on validation loss
    - saves the best checkpoint and training curves
    """
    set_seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUT_DIR / "best_model.pt"
    curve_csv = f"experiments/logs/training_curves/{RUN_NAME}.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Build combined multi-dataset train/validation sets
    train_ds = MultiDatasetFrameDataset("train", make_transforms(train=True), MAX_FRAMES_PER_VIDEO)
    val_ds = MultiDatasetFrameDataset("val", make_transforms(train=False), MAX_FRAMES_PER_VIDEO)

    print("Train frames:", len(train_ds))
    print("Val frames:", len(val_ds))

    # Compute class-balanced sampling weights so minority class samples
    # are drawn more often during training.
    labels = [int(label) for _, label in train_ds.samples]
    class_counts = np.bincount(np.array(labels, dtype=np.int64), minlength=2)
    class_counts = np.maximum(class_counts, 1)
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
        num_workers=NUM_WORKERS,
    )

    # Build pretrained Xception model as the final visual backbone
    model = build_xception_binary(pretrained=True).to(device)

    # Standard binary classification setup
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        # -------------------------
        # Training phase
        # -------------------------
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

        # -------------------------
        # Validation phase
        # -------------------------
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

        # Save epoch-level losses for plotting/reporting
        append_curve_row(curve_csv, {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # Check whether validation loss improved enough to count as progress
        improved = (best_val_loss - val_loss) > MIN_DELTA
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0

            # Save the best checkpoint together with useful metadata
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": float(best_val_loss),
                    "epoch": epoch,
                    "config": {
                        "model": "xception",
                        "img_size": IMG_SIZE,
                        "max_epochs": MAX_EPOCHS,
                        "patience": PATIENCE,
                        "min_delta": MIN_DELTA,
                        "lr": LR,
                        "seed": SEED,
                        "max_frames_per_video": MAX_FRAMES_PER_VIDEO,
                        "datasets": [cfg.name for cfg in DATASETS],
                    },
                },
                best_path,
            )
            print("Saved best ->", best_path)

        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(
                    f"Early stopping: no val_loss improvement > {MIN_DELTA} for {PATIENCE} epoch(s). "
                    f"Best epoch={best_epoch}, best_val_loss={best_val_loss:.4f}"
                )
                break

    print("Done. Best val loss:", best_val_loss)
    print("Best epoch:", best_epoch)
    print("Curves ->", curve_csv)


if __name__ == "__main__":
    main()