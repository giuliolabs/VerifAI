"""
Train Baseline CNN (MobileNetV2) on FaceForensics++ C23 sampled frames
Dependencies:
    pip install torch torchvision torchaudio
    pip install scikit-learn tqdm pillow numpy
Run (from project root):
    python ml/train.py
Outputs:
    experiments/results/ffpp_c23_mobilenet_baseline/best_model.pt
"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ml.data_loader import FFPPFrameDataset
from ml.models.video.mobilenet_baseline import build_mobilenet_v2_binary


def make_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main():
    data_root = Path("data/interim/frames/FaceForensics++_C23")
    out_dir = Path("experiments/results/ffpp_c23_mobilenet_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = FFPPFrameDataset(
        str(train_dir),
        transform=make_transforms(train=True),
        mapping_file=r"data\interim\frames\FaceForensics++_C23\train\_id_map.csv"
    )

    val_ds = FFPPFrameDataset(
        str(val_dir),
        transform=make_transforms(train=False),
        mapping_file=r"data\interim\frames\FaceForensics++_C23\val\_id_map.csv"
    )

    print("Train frames:", len(train_ds))
    print("Val frames:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model = build_mobilenet_v2_binary().to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = 10**9
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, 6):  # 5 epochs baseline
        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for images, labels, video_ids in tqdm(train_loader, desc=f"Epoch {epoch}/5 - train"):
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)

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
            for images, labels, video_ids in tqdm(val_loader, desc=f"Epoch {epoch}/5 - val"):
                images = images.to(device)
                labels = torch.tensor(labels, dtype=torch.float32).to(device)

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
            }, best_path)
            print("Saved best ->", best_path)

    print("Done. Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
