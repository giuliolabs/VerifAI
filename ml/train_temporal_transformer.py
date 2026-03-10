"""
Train Temporal Transformer (T=5) on FF++ C23 Hashed Frames
==========================================================

This script trains the temporal Transformer model on short frame
sequences extracted from the FaceForensics++ C23 dataset. Unlike
frame-level baselines, this model learns from sequences of 5 frames
so it can capture temporal inconsistencies across time.

Run:
  python -m ml.train_temporal_transformer --backbone vit
  python -m ml.train_temporal_transformer --backbone mobilenetv2

Outputs:
  experiments/results/ffpp_c23_temporal_<backbone>/best_model.pt
  experiments/logs/training_curves/temporal_<backbone>.csv

Final training setup:
- max_epochs = 15
- early stopping patience = 3
- validation loss monitored with min_delta
- checkpoint stores only safe serializable types

This script supports two spatial backbones:
- ViT
- MobileNetV2

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path so internal modules can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Dataset for loading fixed-length frame sequences
from ml.video_sequence_data_loader import FFPPSequenceDataset

# Temporal Transformer model builder
from ml.models.video.temporal_transformer import build_temporal_transformer_model

# Utility for saving training curves
from scripts.curve_writer import append_curve_row


# Root folder containing extracted frame sequences
FRAMES_ROOT = Path("data/interim/frames/FaceForensics++_C23")

# Split CSV directory
SPLITS_DIR = Path("data/splits")

TRAIN_CSV = SPLITS_DIR / "faceforensics++_c23_train.csv"
VAL_CSV = SPLITS_DIR / "faceforensics++_c23_val.csv"


def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """
    Main training pipeline for the Temporal Transformer.

    This function:
    - parses the selected backbone
    - loads train/validation sequence datasets
    - applies weighted sampling for class balance
    - trains and validates the model
    - applies early stopping
    - saves the best checkpoint
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mobilenetv2", choices=["vit", "mobilenetv2"])
    args = parser.parse_args()
    backbone = args.backbone

    set_seed(42)

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -------------------------
    # Configuration
    # -------------------------

    # Number of frames per sequence
    t = 5

    # Spatial input size
    img_size = 224

    # Final training settings
    max_epochs = 15
    patience = 3
    min_delta = 1e-4

    # Optimisation settings
    lr = 1e-4
    batch_train = 8
    batch_val = 8
    num_workers = 0

    out_dir = Path(f"experiments/results/ffpp_c23_temporal_{backbone}")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"

    run_name = f"temporal_{backbone}"
    curve_csv = f"experiments/logs/training_curves/{run_name}.csv"

    # -------------------------
    # Data
    # -------------------------

    # Build training dataset
    train_ds = FFPPSequenceDataset(
        frames_root=FRAMES_ROOT,
        split="train",
        split_csv=TRAIN_CSV,
        img_size=img_size,
        t=t,
        train=True,
    )

    # Build validation dataset
    val_ds = FFPPSequenceDataset(
        frames_root=FRAMES_ROOT,
        split="val",
        split_csv=VAL_CSV,
        img_size=img_size,
        t=t,
        train=False,
    )

    print("Train videos:", len(train_ds))
    print("Val videos:", len(val_ds))

    # ---- Weighted sampler (video-level) ----
    # This balances training by sampling videos from minority classes more often.
    labels = [label for _, label in train_ds.samples]
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
        batch_size=batch_train,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_val,
        shuffle=False,
        num_workers=num_workers
    )

    # -------------------------
    # Model
    # -------------------------

    # Build temporal model with chosen spatial backbone
    model = build_temporal_transformer_model(
        backbone=backbone,
        pretrained=True,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        freeze_encoder=False,
    ).to(device)

    # Warmup is only used for ViT:
    # first train higher-level layers briefly, then fine-tune the full model.
    warmup_epochs = 2
    finetune_lr = lr * 0.1

    if backbone == "vit":
        if hasattr(model, "encoder"):
            for p in model.encoder.parameters():
                p.requires_grad = False
    else:
        # MobileNetV2 is lightweight enough to fine-tune directly
        warmup_epochs = 0
        finetune_lr = lr

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # -------------------------
    # Early stopping state
    # -------------------------
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, max_epochs + 1):

        # For ViT, unfreeze encoder after warmup and continue with smaller LR
        if backbone == "vit" and epoch == warmup_epochs + 1:
            if hasattr(model, "encoder"):
                for p in model.encoder.parameters():
                    p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)

        # ---- Training phase ----
        model.train()
        train_loss_sum, train_count = 0.0, 0

        for frames, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} - train"):
            frames = frames.to(device)         # [B, T, 3, H, W]
            y = y.float().to(device)           # [B]

            logits = model(frames).squeeze(1)  # [B]
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = frames.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)

        # ---- Validation phase ----
        model.eval()
        val_loss_sum, val_count = 0.0, 0

        with torch.no_grad():
            for frames, y, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{max_epochs} - val"):
                frames = frames.to(device)
                y = y.float().to(device)

                logits = model(frames).squeeze(1)
                loss = loss_fn(logits, y)

                bs = frames.size(0)
                val_loss_sum += float(loss.item()) * bs
                val_count += bs

        val_loss = val_loss_sum / max(val_count, 1)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Save training curve row after each epoch
        append_curve_row(curve_csv, {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # ---- Early stopping + checkpoint ----
        improved = (best_val_loss - val_loss) > min_delta

        if improved:
            best_val_loss = val_loss
            no_improve = 0

            # Save only tensors and plain Python values to keep
            # checkpoints safe and easier to reload across environments.
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": int(epoch),
                    "val_loss": float(best_val_loss),
                    "config": {
                        "backbone": str(backbone),
                        "t": int(t),
                        "img_size": int(img_size),
                        "max_epochs": int(max_epochs),
                        "patience": int(patience),
                        "min_delta": float(min_delta),
                        "lr": float(lr),
                        "finetune_lr": float(finetune_lr),
                        "warmup_epochs": int(warmup_epochs),
                        "sampler": "WeightedRandomSampler",
                        "batch_train": int(batch_train),
                        "batch_val": int(batch_val),
                        "num_workers": int(num_workers),
                    },
                },
                best_path,
            )
            print("Saved best ->", best_path)

        else:
            no_improve += 1
            if no_improve >= patience:
                break

    print("Done. Best val loss:", best_val_loss)
    print("Curves ->", curve_csv)


if __name__ == "__main__":
    main()