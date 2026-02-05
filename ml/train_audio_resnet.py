"""
Train Audio-Only Model: MFCC -> ResNet18 (binary).

Dataset: FakeAVCeleb_v1.2 (audio available)
Input: MFCC .npy files (n_mfcc=40)

Dependencies:
    pip install torch torchvision
    pip install numpy pandas scikit-learn tqdm

Run (from project root):
    python -m ml.train_audio_resnet

Outputs:
    experiments/results/fakeavceleb_audio_resnet_baseline/best_model.pt
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.audio_data_loader import MFCCDataset
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary


def main():
    train_csv = "data/splits/fakeavceleb_train.csv"
    val_csv   = "data/splits/fakeavceleb_val.csv"

    mfcc_train_root = "data/processed/audio_features/FakeAVCeleb_v1.2/train"
    mfcc_val_root   = "data/processed/audio_features/FakeAVCeleb_v1.2/val"

    out_dir = Path("experiments/results/fakeavceleb_audio_resnet_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = MFCCDataset(train_csv, mfcc_train_root)
    val_ds   = MFCCDataset(val_csv, mfcc_val_root)

    print("Train items:", len(train_ds))
    print("Val items:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = build_mfcc_resnet18_binary(pretrained=True).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = 10**9

    for epoch in range(1, 6):  # 5 epoch baseline
        # ---- train ----
        model.train()
        train_loss_sum, train_count = 0.0, 0

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/5 - train"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)

        # ---- val ----
        model.eval()
        val_loss_sum, val_count = 0.0, 0

        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc=f"Epoch {epoch}/5 - val"):
                x = x.to(device)
                y = y.to(device)

                logits = model(x).squeeze(1)
                loss = loss_fn(logits, y)

                bs = x.size(0)
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
            }, best_path)
            print("Saved best ->", best_path)

    print("Done. Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
