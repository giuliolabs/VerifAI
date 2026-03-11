"""
Train Baseline CNN (MobileNetV2) on FaceForensics++ C23 Sampled Frames
======================================================================

This script trains the MobileNetV2 visual baseline on sampled frames
from the FaceForensics++ C23 dataset. It is intended as a lightweight
benchmark model for frame-level deepfake detection.

Upgrades:
- max_epochs increased for longer training
- early stopping added based on validation loss
- best_model.pt still saved whenever validation improves

Run (from project root):
    python -m ml.train

This baseline is useful because MobileNetV2 is computationally efficient,
making it suitable for quick experiments, comparisons, and ablation work.

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

# Project-specific transforms, dataset loader, and model builder
from ml.augmentations import get_transforms
from ml.data_loader import FFPPFrameDataset
from ml.models.video.mobilenet_baseline import build_mobilenet_v2_binary


def main():
    """
    Main training pipeline for the MobileNetV2 baseline.

    This function:
    - loads the train/validation frame datasets
    - builds the MobileNetV2 binary classifier
    - applies class weighting to reduce class imbalance bias
    - trains and validates the model
    - uses early stopping based on validation loss
    - saves the best checkpoint
    """

    # Root dataset folder and output checkpoint folder
    data_root = Path("data/interim/frames/FaceForensics++_C23")
    out_dir = Path("experiments/results/ffpp_c23_mobilenet_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- Training control ----
    # These settings control maximum training duration and early stopping behavior
    max_epochs = 15
    patience = 3
    min_delta = 1e-4  # minimum improvement required in validation loss

    # Build training dataset with augmentation enabled
    train_ds = FFPPFrameDataset(
        str(train_dir),
        transform=get_transforms(train=True),
        mapping_file=r"data\interim\frames\FaceForensics++_C23\train\_id_map.csv"
    )

    # Build validation dataset with deterministic transforms only
    val_ds = FFPPFrameDataset(
        str(val_dir),
        transform=get_transforms(train=False),
        mapping_file=r"data\interim\frames\FaceForensics++_C23\val\_id_map.csv"
    )

    print("Train frames:", len(train_ds))
    print("Val frames:", len(val_ds))

    # DataLoaders handle batching and shuffling
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    # Build MobileNetV2 binary classifier and move it to the selected device
    model = build_mobilenet_v2_binary().to(device)

    # Count training labels to estimate class imbalance
    num_fake = sum(1 for _, y, _ in train_ds.items if y == 1)
    num_real = sum(1 for _, y, _ in train_ds.items if y == 0)

    # Weight the positive class so the model is less likely to favor
    # the majority class when the dataset is imbalanced.
    pos_weight = torch.tensor([num_real / max(num_fake, 1)], device=device)

    # BCEWithLogitsLoss is appropriate for single-logit binary classification
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # AdamW provides stable optimization with mild regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    best_val_loss = float("inf")
    best_path = out_dir / "best_model.pt"

    # Counter used for early stopping
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # ---- Training phase ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for images, labels, _video_ids in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} - train"):
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)

        # ---- Validation phase ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for images, labels, _video_ids in tqdm(val_loader, desc=f"Epoch {epoch}/{max_epochs} - val"):
                images = images.to(device)

                # Labels are already tensors from the DataLoader,
                # so converting directly to float avoids unnecessary warnings
                labels = labels.float().to(device)

                logits = model(images).squeeze(1)
                loss = loss_fn(logits, labels)

                bs = images.size(0)
                val_loss_sum += loss.item() * bs
                val_count += bs

        val_loss = val_loss_sum / max(val_count, 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ---- Early stopping + checkpointing ----
        # Validation loss is used as the main model-selection criterion
        improved = (best_val_loss - val_loss) > min_delta

        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save checkpoint whenever validation loss improves sufficiently
            torch.save({
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "epoch": epoch,
                "max_epochs": max_epochs,
                "patience": patience,
            }, best_path)
            print("Saved best ->", best_path)

        else:
            epochs_no_improve += 1

            # Stop early if the model has not improved for several epochs
            if epochs_no_improve >= patience:
                break

    print("Done. Best val loss:", best_val_loss)
    print("Best checkpoint ->", best_path)


if __name__ == "__main__":
    main()