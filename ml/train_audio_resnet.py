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
    experiments/logs/training_curves/audio_resnet_baseline.csv
    experiments/logs/runs/run_*.json
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from ml.audio_data_loader import MFCCDataset
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary

from scripts.curve_writer import append_curve_row
from scripts.run_logger import log_run, log_exception


def main():
    run_name = "audio_resnet_baseline"
    curve_csv = f"experiments/logs/training_curves/{run_name}.csv"

    try:
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

        if len(train_ds) == 0 or len(val_ds) == 0:
            raise FileNotFoundError(
                "No MFCC samples found. Check your split CSVs and mfcc_*_root paths."
            )

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        model = build_mfcc_resnet18_binary(pretrained=True).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        best_val_loss = float("inf")
        best_epoch = 0
        best_val_auc = float("nan")
        best_val_f1 = float("nan")

        for epoch in range(1, 6):  # 5 epoch baseline
            # ---- train ----
            model.train()
            train_loss_sum, train_count = 0.0, 0

            for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/5 - train"):
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

            # ---- val ----
            model.eval()
            val_loss_sum, val_count = 0.0, 0
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for x, y, _ in tqdm(val_loader, desc=f"Epoch {epoch}/5 - val"):
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

            val_auc = float("nan")
            val_f1 = float("nan")

            if len(set(all_labels)) > 1:
                val_auc = roc_auc_score(all_labels, all_probs)

            preds = (np.array(all_probs) >= 0.5).astype(int).tolist()
            val_f1 = f1_score(all_labels, preds, zero_division=0)

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_auc={val_auc:.4f}  val_f1={val_f1:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_auc = val_auc
                best_val_f1 = val_f1
                best_epoch = epoch

                torch.save({
                    "model_state": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_auc": best_val_auc,
                    "val_f1": best_val_f1,
                    "epoch": best_epoch,
                    "config": {
                        "model": "mfcc_resnet18",
                        "lr": 1e-4,
                        "epochs": 5,
                        "batch_train": 32,
                        "batch_val": 64,
                    }
                }, best_path)
                print("Saved best ->", best_path)

            append_curve_row(curve_csv, {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_f1": val_f1,
            })

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
            "notes": "Baseline 5 epochs. Curves + AUC/F1 logged.",
        })

        print("Done. Best val loss:", best_val_loss)

    except Exception as exc:
        log_exception(run_name, exc, context={"script": "ml/train_audio_resnet.py"})
        raise


if __name__ == "__main__":
    main()
