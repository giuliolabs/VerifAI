"""
Train Visual CNN (Xception) on FaceForensics++ C23 sampled frames.

Dependencies:
    pip install torch torchvision timm
    pip install scikit-learn tqdm pillow numpy

Run (from project root):
    python -m ml.train_xception

Outputs:
    experiments/results/ffpp_c23_xception_baseline/best_model.pt
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ml.data_loader import FFPPFrameDataset
from ml.models.video.xception import build_xception_binary


# ==============================
# CONFIG
# ==============================
DATA_ROOT = Path("data/interim/frames/FaceForensics++_C23")
OUT_DIR = Path("experiments/results/ffpp_c23_xception_baseline")

IMG_SIZE = 299
BATCH_TRAIN = 16
BATCH_VAL = 32
EPOCHS = 5
LR = 1e-4
NUM_WORKERS = 2
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(train: bool):
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


def main():
    set_seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUT_DIR / "best_model.pt"

    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = FFPPFrameDataset(str(train_dir), transform=make_transforms(train=True))
    val_ds = FFPPFrameDataset(str(val_dir), transform=make_transforms(train=False))

    print("Train frames:", len(train_ds))
    print("Val frames:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False, num_workers=NUM_WORKERS)

    model = build_xception_binary(pretrained=True).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for images, labels, video_ids in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - train"):
            images = images.to(device)
            labels = labels.float().to(device)  # <-- removes warning

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
            for images, labels, video_ids in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - val"):
                images = images.to(device)
                labels = labels.float().to(device)

                logits = model(images).squeeze(1)
                loss = loss_fn(logits, labels)

                bs = images.size(0)
                val_loss_sum += loss.item() * bs
                val_count += bs

        val_loss = val_loss_sum / max(val_count, 1)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "epoch": epoch,
                "config": {
                    "model": "xception",
                    "img_size": IMG_SIZE,
                    "epochs": EPOCHS,
                    "lr": LR,
                    "seed": SEED,
                }
            }, best_path)
            print("Saved best ->", best_path)

    print("Done. Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
