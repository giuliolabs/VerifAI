"""
Train Audio-Only Model: MFCC -> ResNet18 (Binary)
=================================================

This script trains the MFCC-based audio baseline used in VerifAI.
It takes precomputed MFCC feature tensors and trains a ResNet18 model
adapted for binary classification (real vs fake).

Dataset:
    FakeAVCeleb_v1.2 (audio available)

Input:
    MFCC .npy files (n_mfcc=40)

Dependencies:
    pip install torch torchvision
    pip install numpy pandas scikit-learn tqdm

Run (from project root):
    python -m ml.train_audio_resnet

Outputs:
    experiments/results/fakeavceleb_audio_resnet_baseline/best_model.pt
    experiments/logs/training_curves/audio_resnet_baseline.csv
    experiments/logs/runs/run_*.json

This baseline is important because it provides the final audio backbone
later reused by the hybrid audio-visual fusion system.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Add project root to Python path so internal modules can be imported
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# PyTorch core modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# tqdm provides progress bars during training and validation
from tqdm import tqdm

# NumPy is used for thresholding and metric preparation
import numpy as np

# Validation metrics
from sklearn.metrics import roc_auc_score, f1_score

# Project dataset and model builder
from ml.audio_data_loader import MFCCDataset
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary

# Logging utilities for training curves and run/error tracking
from scripts.curve_writer import append_curve_row
from scripts.run_logger import log_run, log_exception


def main():
    """
    Main training pipeline for the MFCC-based audio baseline.

    This function:
    - loads train/validation MFCC datasets
    - trains the ResNet18 audio model
    - evaluates validation performance after each epoch
    - saves the best checkpoint based on validation loss
    - applies early stopping
    - writes central VerifAI logs
    """
    run_name = "audio_resnet_baseline"
    curve_csv = f"experiments/logs/training_curves/{run_name}.csv"

    # ---------------------------
    # Training policy
    # ---------------------------

    # Maximum number of training epochs
    MAX_EPOCHS = 15

    # Stop early if validation loss stops improving
    PATIENCE = 3

    # Ignore tiny validation-loss changes
    MIN_DELTA = 1e-4

    try:
        # Paths to train/validation split CSV files
        train_csv = "data/splits/fakeavceleb_train.csv"
        val_csv = "data/splits/fakeavceleb_val.csv"

        # Root folders containing precomputed MFCC features
        mfcc_train_root = "data/processed/audio_features/FakeAVCeleb_v1.2/train"
        mfcc_val_root = "data/processed/audio_features/FakeAVCeleb_v1.2/val"

        # Output folder for best checkpoint
        out_dir = Path("experiments/results/fakeavceleb_audio_resnet_baseline")
        out_dir.mkdir(parents=True, exist_ok=True)
        best_path = out_dir / "best_model.pt"

        # Select GPU if available, otherwise use CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", device)

        # ---------------------------
        # Data
        # ---------------------------

        # Build MFCC train/validation datasets
        train_ds = MFCCDataset(train_csv, mfcc_train_root)
        val_ds = MFCCDataset(val_csv, mfcc_val_root)

        print("Train items:", len(train_ds))
        print("Val items:", len(val_ds))

        # Fail early if MFCC features are missing or dataset setup is incomplete
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise FileNotFoundError(
                "No MFCC samples found. Check your split CSVs and mfcc_*_root paths."
            )

        # DataLoaders handle batching and shuffling
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        # ---------------------------
        # Model
        # ---------------------------

        # Build ResNet18 adapted for MFCC input
        model = build_mfcc_resnet18_binary(pretrained=True).to(device)

        # BCEWithLogitsLoss is appropriate because the model outputs one binary logit
        loss_fn = nn.BCEWithLogitsLoss()

        # AdamW is used for stable optimization and mild regularization
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        best_val_loss = float("inf")
        best_epoch = 0
        best_val_auc = float("nan")
        best_val_f1 = float("nan")

        # Early stopping counter
        epochs_no_improve = 0

        for epoch in range(1, MAX_EPOCHS + 1):

            # ---------------------------
            # Training phase
            # ---------------------------
            model.train()
            train_loss_sum, train_count = 0.0, 0

            for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} - train"):
                x = x.to(device)
                y = y.float().to(device)

                logits = model(x).squeeze(1)
                loss = loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                train_loss_sum += loss.item() * bs
                train_count += bs

            train_loss = train_loss_sum / max(train_count, 1)

            # ---------------------------
            # Validation phase
            # ---------------------------
            model.eval()
            val_loss_sum, val_count = 0.0, 0
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for x, y, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} - val"):
                    x = x.to(device)
                    y = y.float().to(device)

                    logits = model(x).squeeze(1)
                    loss = loss_fn(logits, y)

                    probs = torch.sigmoid(logits)

                    bs = x.size(0)
                    val_loss_sum += loss.item() * bs
                    val_count += bs

                    all_labels.extend(y.detach().cpu().numpy().tolist())
                    all_probs.extend(probs.detach().cpu().numpy().tolist())

            val_loss = val_loss_sum / max(val_count, 1)

            # Compute validation AUC only if both classes are present
            val_auc = float("nan")
            if len(set(all_labels)) > 1:
                val_auc = roc_auc_score(all_labels, all_probs)

            # Convert probabilities to binary predictions for F1 score
            preds = (np.array(all_probs) >= 0.5).astype(int).tolist()
            val_f1 = f1_score(all_labels, preds, zero_division=0)

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_auc={val_auc:.4f}  val_f1={val_f1:.4f}"
            )

            # ---------------------------
            # Checkpointing + early stopping
            # ---------------------------

            # Validation loss is treated as the main model-selection metric
            improved = (best_val_loss - val_loss) > MIN_DELTA

            if improved:
                best_val_loss = val_loss
                best_val_auc = val_auc
                best_val_f1 = val_f1
                best_epoch = epoch
                epochs_no_improve = 0

                # Save the best checkpoint with model weights and key training metadata
                torch.save({
                    "model_state": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_auc": best_val_auc,
                    "val_f1": best_val_f1,
                    "epoch": best_epoch,
                    "config": {
                        "model": "mfcc_resnet18",
                        "lr": 1e-4,
                        "epochs": MAX_EPOCHS,
                        "patience": PATIENCE,
                        "min_delta": MIN_DELTA,
                        "batch_train": 32,
                        "batch_val": 64,
                    }
                }, best_path)
                print("Saved best ->", best_path)

            else:
                epochs_no_improve += 1

                if epochs_no_improve >= PATIENCE:
                    print(
                        f"Early stopping: no val_loss improvement > {MIN_DELTA} "
                        f"for {PATIENCE} epoch(s). Best epoch={best_epoch}, best_val_loss={best_val_loss:.4f}"
                    )

                    # Still save the final epoch metrics before stopping
                    append_curve_row(curve_csv, {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_auc": val_auc,
                        "val_f1": val_f1,
                    })
                    break

            # Save epoch-level metrics for plotting and reporting
            append_curve_row(curve_csv, {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_f1": val_f1,
            })

        # Write final experiment summary to the central VerifAI run log
        log_run(run_name, payload={
            "task": "train",
            "model": "mfcc_resnet18",
            "dataset": "FakeAVCeleb_v1.2",
            "split": "train/val",
            "seed": None,
            "metrics": {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_val_auc": best_val_auc,
                "best_val_f1": best_val_f1,
            },
            "artifacts": [str(best_path), str(curve_csv)],
            "notes": f"Training with early stopping. max_epochs={MAX_EPOCHS}, patience={PATIENCE}, min_delta={MIN_DELTA}.",
        })

        print("Done. Best val loss:", best_val_loss)

    except Exception as exc:
        # Log unexpected failures before re-raising the exception
        log_exception(run_name, exc, context={"script": "ml/train_audio_resnet.py"})
        raise


if __name__ == "__main__":
    main()