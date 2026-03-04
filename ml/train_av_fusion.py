"""
Train Multimodal Fusion Model (Audio + Video) on FakeAVCeleb_v1.2

What this script does
---------------------
- Loads multimodal samples using ml/av_data_loader.FakeAVCelebAVDataset
- Trains a fusion model that combines:
    - video frames tensor: [B, 5, 3, 224, 224]
    - MFCC tensor:         [B, 1, 40, T]  (T fixed via mfcc_max_len)
- Saves best checkpoint (lowest val_loss)
- Writes fusion_log.csv with loss + metrics + embedding norm statistics
- Writes VerifAI central logs:
    - experiments/logs/training_curves/av_fusion_v1.csv
    - experiments/logs/runs/run_*.json
    - experiments/logs/errors/error_*.json

Dependencies:
    pip install torch torchvision numpy pandas scikit-learn tqdm pillow

Run (from project root):
    python -m ml.train_av_fusion

Outputs:
    experiments/results/fakeavceleb_av_fusion_v1/best_model.pt
    experiments/results/fakeavceleb_av_fusion_v1/fusion_log.csv

Notes:
- IMPORTANT: set mfcc_max_len to make batching possible.
- Uses num_workers=0 by default (more stable on Windows/OneDrive).
  You can increase to 2 if stable.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from ml.av_data_loader import FakeAVCelebAVDataset
from ml.models.fusion.multimodal_fusion import MultimodalFusionModel

from scripts.curve_writer import append_curve_row
from scripts.run_logger import log_run, log_exception


def _to_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    # supports [B], [B,1]
    if logits.ndim == 2 and logits.size(1) == 1:
        return logits.squeeze(1)
    return logits


def evaluate(model, loader, device) -> tuple[float, float, float, float, float, float]:
    """
    Returns:
        val_loss, acc, f1, auc, (video_norm_mean, audio_norm_mean)
    Assumes model optionally exposes last embedding norms:
        model.last_video_norm_mean
        model.last_audio_norm_mean
    If not present, returns NaN for those.
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    y_true = []
    y_prob = []
    loss_sum = 0.0
    count = 0

    video_norms = []
    audio_norms = []

    with torch.no_grad():
        for video, mfcc, y, _ in tqdm(loader, desc="val", leave=False):
            video = video.to(device)
            mfcc = mfcc.to(device)
            y = y.float().to(device)

            logits = model(video, mfcc)
            logits = _to_1d_logits(logits)

            loss = loss_fn(logits, y)
            bs = y.size(0)
            loss_sum += float(loss.item()) * bs
            count += bs

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(y.detach().cpu().numpy().astype(int).tolist())

            # Optional fusion logging (if model provides these)
            if hasattr(model, "last_video_norm_mean"):
                vn = getattr(model, "last_video_norm_mean")
                if vn is not None:
                    video_norms.append(float(vn))
            if hasattr(model, "last_audio_norm_mean"):
                an = getattr(model, "last_audio_norm_mean")
                if an is not None:
                    audio_norms.append(float(an))

    val_loss = loss_sum / max(count, 1)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    video_norm_mean = float(np.mean(video_norms)) if video_norms else float("nan")
    audio_norm_mean = float(np.mean(audio_norms)) if audio_norms else float("nan")

    return val_loss, acc, f1, auc, video_norm_mean, audio_norm_mean


def main():
    run_name = "av_fusion_v1"
    curve_csv = f"experiments/logs/training_curves/{run_name}.csv"

    try:
        # -------------------------
        # Config
        # -------------------------
        out_dir = Path("experiments/results/fakeavceleb_av_fusion_v1")
        out_dir.mkdir(parents=True, exist_ok=True)

        best_path = out_dir / "best_model.pt"
        log_path = out_dir / "fusion_log.csv"

        MAX_EPOCHS = 15
        PATIENCE = 3
        MIN_DELTA = 1e-4

        lr = 1e-4
        batch_size = 8
        num_workers = 0
        mfcc_max_len = 200

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", device)

        # -------------------------
        # Data
        # -------------------------
        train_ds = FakeAVCelebAVDataset("train", strict=True, mfcc_max_len=mfcc_max_len)
        val_ds   = FakeAVCelebAVDataset("val",   strict=True, mfcc_max_len=mfcc_max_len)

        print("Train items:", len(train_ds))
        print("Val items:", len(val_ds))

        if len(train_ds) == 0 or len(val_ds) == 0:
            raise FileNotFoundError(
                "No AV samples found. Check FakeAVCeleb paths, splits, and preprocessing outputs."
            )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # -------------------------
        # Model
        # -------------------------
        model = MultimodalFusionModel().to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # -------------------------
        # Logging setup
        # -------------------------
        if not log_path.exists():
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_acc",
                    "val_f1",
                    "val_auc",
                    "video_emb_norm_mean",
                    "audio_emb_norm_mean",
                ])

        best_val_loss = float("inf")
        best_epoch = 0
        best_val_acc = float("nan")
        best_val_f1 = float("nan")
        best_val_auc = float("nan")

        epochs_no_improve = 0

        # -------------------------
        # Train loop
        # -------------------------
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for video, mfcc, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} - train"):
                video = video.to(device)
                mfcc = mfcc.to(device)
                y = y.float().to(device)

                logits = model(video, mfcc)
                logits = _to_1d_logits(logits)

                loss = loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                train_loss_sum += float(loss.item()) * bs
                train_count += bs

            train_loss = train_loss_sum / max(train_count, 1)

            # -------------------------
            # Validation
            # -------------------------
            val_loss, val_acc, val_f1, val_auc, vnorm, anorm = evaluate(model, val_loader, device)

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"acc={val_acc:.4f}  f1={val_f1:.4f}  auc={val_auc:.4f}"
            )

            # write fusion log
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, val_acc, val_f1, val_auc, vnorm, anorm])

            append_curve_row(curve_csv, {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_acc": val_acc,
            })

            improved = (best_val_loss - val_loss) > MIN_DELTA

            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_auc = val_auc
                epochs_no_improve = 0

                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": best_epoch,
                    "val_loss": best_val_loss,
                    "val_acc": best_val_acc,
                    "val_f1": best_val_f1,
                    "val_auc": best_val_auc,
                    "mfcc_max_len": mfcc_max_len,
                    "config": {
                        "model": "multimodal_fusion",
                        "epochs": MAX_EPOCHS,
                        "patience": PATIENCE,
                        "lr": lr,
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "mfcc_max_len": mfcc_max_len,
                    }
                }, best_path)

                print("Saved best ->", best_path)

            else:
                epochs_no_improve += 1

                if epochs_no_improve >= PATIENCE:
                    print(
                        f"Early stopping triggered. "
                        f"Best epoch={best_epoch} | best_val_loss={best_val_loss:.4f}"
                    )
                    break

        log_run(run_name, payload={
            "task": "train",
            "model": "multimodal_fusion",
            "dataset": "FakeAVCeleb_v1.2",
            "split": "train/val",
            "metrics": {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "best_val_auc": best_val_auc,
                "best_val_f1": best_val_f1,
            },
            "artifacts": [str(best_path), str(log_path), str(curve_csv)],
            "notes": "Fusion baseline 5 epochs. Logs written to fusion_log.csv + central VerifAI logs.",
        })

        print("Done. Best val loss:", best_val_loss)
        print("Fusion log ->", log_path)

    except Exception as exc:
        log_exception(run_name, exc, context={"script": "ml/train_av_fusion.py"})
        raise


if __name__ == "__main__":
    main()
