"""
Train Audio-Only Model: Raw WAV -> 1D CNN Encoder (Binary)
==========================================================

This script trains the raw-waveform audio baseline used in VerifAI.
It takes fixed-length .wav inputs from the FakeAVCeleb dataset and
trains a lightweight 1D CNN to perform binary classification
(real vs fake).

Dataset:
    FakeAVCeleb_v1.2 (audio)

Input:
    .wav files under:
    data/interim/audio/FakeAVCeleb_v1.2/<split>/

Outputs:
    experiments/results/fakeavceleb_wav_encoder_baseline/best_model.pt
    experiments/logs/training_curves/wav_encoder_baseline.csv

This experiment is useful as an ablation against MFCC-based audio models,
because it tests whether the system can learn directly from raw waveforms
without handcrafted audio features.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import sys
from pathlib import Path
import random

# Add project root to Python path so internal modules can be imported cleanly
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Numerical processing
import numpy as np

# PyTorch core modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# tqdm provides progress bars during training and validation
from tqdm import tqdm

# Project imports
from ml.wav_data_loader import WavDataset
from ml.models.audio.wav_encoder import build_wav_binary_classifier
from scripts.curve_writer import append_curve_row


# -------------------------
# Configuration
# -------------------------

# CSV split files defining train and validation samples
TRAIN_CSV = "data/splits/fakeavceleb_train.csv"
VAL_CSV   = "data/splits/fakeavceleb_val.csv"

# Root folders containing extracted WAV files
WAV_TRAIN_ROOT = "data/interim/audio/FakeAVCeleb_v1.2/train"
WAV_VAL_ROOT   = "data/interim/audio/FakeAVCeleb_v1.2/val"

# Output directory for checkpoint
OUT_DIR  = Path("experiments/results/fakeavceleb_wav_encoder_baseline")

# Run name used for logging/training curves
RUN_NAME = "wav_encoder_baseline"

# Project-wide training defaults
MAX_EPOCHS = 15
PATIENCE   = 3
MIN_DELTA  = 1e-4

# Optimisation settings
LR = 1e-4
BATCH_TRAIN = 16
BATCH_VAL   = 32
NUM_WORKERS = 0
SEED = 42

# WAV preprocessing settings
SAMPLE_RATE = 16000
SECONDS = 3

# If True, dataset enforces exact sample rate matching.
# Here it is disabled to make the data loader slightly more forgiving.
STRICT_SR = False


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    This helps keep training behavior more consistent across runs,
    especially for dataset shuffling and parameter initialization.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """
    Main training pipeline for the raw-waveform audio baseline.

    This function:
    - loads train/validation datasets
    - builds the 1D CNN classifier
    - trains and validates the model
    - applies early stopping
    - saves the best checkpoint
    - logs training curves
    """
    set_seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUT_DIR / "best_model.pt"

    # Select GPU if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -------------------------
    # Data
    # -------------------------

    # Build training dataset
    train_ds = WavDataset(
        split_csv=TRAIN_CSV,
        wav_root=WAV_TRAIN_ROOT,
        sample_rate=SAMPLE_RATE,
        seconds=SECONDS,
        strict_sr=STRICT_SR,
    )

    # Build validation dataset
    val_ds = WavDataset(
        split_csv=VAL_CSV,
        wav_root=WAV_VAL_ROOT,
        sample_rate=SAMPLE_RATE,
        seconds=SECONDS,
        strict_sr=STRICT_SR,
    )

    print("Train items:", len(train_ds))
    print("Val items:", len(val_ds))

    # DataLoaders handle batching and shuffling.
    # pin_memory is enabled when CUDA is available to speed up host-to-device transfer.
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_VAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------
    # Model
    # -------------------------

    # Build raw-waveform binary classifier.
    # The encoder learns directly from waveform input rather than MFCC features.
    model = build_wav_binary_classifier(
        sample_rate=SAMPLE_RATE,
        embedding_dim=256,
        base_channels=32,
        dropout=0.2,
    ).to(device)

    # BCEWithLogitsLoss is appropriate because the model outputs a single binary logit
    loss_fn = nn.BCEWithLogitsLoss()

    # AdamW is used for stable optimization and regularized weight updates
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    bad_epochs = 0

    curve_csv = f"experiments/logs/training_curves/{RUN_NAME}.csv"

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, MAX_EPOCHS + 1):

        # ---- Training phase ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} - train"):
            x = x.to(device)
            y = y.float().to(device)

            # Forward pass
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)

            # Standard optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)

        # ---- Validation phase ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} - val"):
                x = x.to(device)
                y = y.float().to(device)

                logits = model(x).squeeze(1)
                loss = loss_fn(logits, y)

                bs = x.size(0)
                val_loss_sum += float(loss.item()) * bs
                val_count += bs

        val_loss = val_loss_sum / max(val_count, 1)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ---- Early stopping + checkpointing ----

        # Improvement is only counted if validation loss decreases
        # by more than MIN_DELTA. This avoids saving models for tiny fluctuations.
        improved = (best_val_loss - val_loss) > MIN_DELTA

        if improved:
            best_val_loss = val_loss
            bad_epochs = 0

            # Save the best checkpoint together with useful configuration metadata
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": float(best_val_loss),
                    "epoch": epoch,
                    "config": {
                        "model": "wav_encoder_1d",
                        "sample_rate": SAMPLE_RATE,
                        "seconds": SECONDS,
                        "fixed_samples": SAMPLE_RATE * SECONDS,
                        "embedding_dim": 256,
                        "base_channels": 32,
                        "dropout": 0.2,
                        "max_epochs": MAX_EPOCHS,
                        "patience": PATIENCE,
                        "min_delta": MIN_DELTA,
                        "lr": LR,
                        "seed": SEED,
                    },
                },
                best_path,
            )
            print("Saved best ->", best_path)

        else:
            bad_epochs += 1

            # Stop training early if validation performance stops improving
            if bad_epochs >= PATIENCE:
                break

        # ---- Training curve logging ----

        # Save epoch-level losses for later plotting/reporting
        append_curve_row(
            curve_csv,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        )

    print("Done. Best val loss:", best_val_loss)
    print("Curves CSV ->", curve_csv)


if __name__ == "__main__":
    main()