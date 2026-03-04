"""
Train Baseline CNN (MobileNetV2) on FaceForensics++ C23 sampled frames

Upgrades:
- max_epochs = 30
- early stopping with patience = 5 (monitors val_loss)
- still saves best_model.pt

Run (from project root):
    python -m ml.train
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.augmentations import get_transforms
from ml.data_loader import FFPPFrameDataset
from ml.models.video.mobilenet_baseline import build_mobilenet_v2_binary


def main():
    data_root = Path("data/interim/frames/FaceForensics++_C23")
    out_dir = Path("experiments/results/ffpp_c23_mobilenet_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- training control ----
    max_epochs = 15
    patience = 3
    min_delta = 1e-4  # require at least this much improvement in val_loss

    train_ds = FFPPFrameDataset(
        str(train_dir),
        transform=get_transforms(train=True),
        mapping_file=r"data\interim\frames\FaceForensics++_C23\train\_id_map.csv"
    )

    val_ds = FFPPFrameDataset(
        str(val_dir),
        transform=get_transforms(train=False),
        mapping_file=r"data\interim\frames\FaceForensics++_C23\val\_id_map.csv"
    )

    print("Train frames:", len(train_ds))
    print("Val frames:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model = build_mobilenet_v2_binary().to(device)

    # count labels in training set
    num_fake = sum(1 for _, y, _ in train_ds.items if y == 1)
    num_real = sum(1 for _, y, _ in train_ds.items if y == 0)

    # weight positives so the model doesn't just predict the majority class
    pos_weight = torch.tensor([num_real / max(num_fake, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    best_val_loss = float("inf")
    best_path = out_dir / "best_model.pt"

    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # ---- train ----
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

        # ---- val ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for images, labels, _video_ids in tqdm(val_loader, desc=f"Epoch {epoch}/{max_epochs} - val"):
                images = images.to(device)
                labels = labels.float().to(device)  # <-- FIX: avoid torch.tensor(labels) warning

                logits = model(images).squeeze(1)
                loss = loss_fn(logits, labels)

                bs = images.size(0)
                val_loss_sum += loss.item() * bs
                val_count += bs

        val_loss = val_loss_sum / max(val_count, 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ---- early stopping + checkpoint ----
        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0

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

            if epochs_no_improve >= patience:
                break

    print("Done. Best val loss:", best_val_loss)
    print("Best checkpoint ->", best_path)


if __name__ == "__main__":
    main()
