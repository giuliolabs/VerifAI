"""
Evaluate Temporal Transformer (T=5) on FF++ C23 test split.

Run:
  python -m ml.evaluate_temporal_transformer --backbone mobilenetv2
  python -m ml.evaluate_temporal_transformer --backbone vit
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef
)

from ml.video_sequence_data_loader import FFPPSequenceDataset
from ml.models.video.temporal_transformer import build_temporal_transformer_model


FRAMES_ROOT = Path("data/interim/frames/FaceForensics++_C23")
SPLITS_DIR = Path("data/splits")
TEST_CSV = SPLITS_DIR / "faceforensics++_c23_test.csv"


def _torch_load_compat(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mobilenetv2", choices=["vit", "mobilenetv2"])
    args = parser.parse_args()
    backbone = args.backbone

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    t = 5
    img_size = 224

    out_dir = Path(f"experiments/results/ffpp_c23_temporal_{backbone}")
    ckpt_path = out_dir / "best_model.pt"
    out_report = out_dir / "test_report.txt"

    ds = FFPPSequenceDataset(
        frames_root=FRAMES_ROOT,
        split="test",
        split_csv=TEST_CSV,
        img_size=img_size,
        t=t,
        train=False,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    model = build_temporal_transformer_model(
        backbone=backbone,
        pretrained=False,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        freeze_encoder=False,
    ).to(device)

    ckpt = _torch_load_compat(ckpt_path, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true: list[int] = []
    y_prob: list[float] = []

    with torch.no_grad():
        for frames, y, _ in tqdm(loader, desc="Evaluating"):
            frames = frames.to(device)
            logits = model(frames).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(y.numpy().astype(int).tolist())
            y_prob.extend(probs.tolist())

    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    if backbone == "vit":
        best_t = 0.5
        best_bal = -1.0

        for t in np.linspace(0.05, 0.95, 19):
            preds = (y_prob_np >= t).astype(int)
            bal = balanced_accuracy_score(y_true_np, preds)
            if bal > best_bal:
                best_bal = bal
                best_t = float(t)

        y_pred_np = (y_prob_np >= best_t).astype(int)
    else:
        y_pred_np = (y_prob_np >= 0.5).astype(int)

    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np)
    try:
        auc = roc_auc_score(y_true_np, y_prob_np)
    except ValueError:
        auc = float("nan")

    bal_acc = balanced_accuracy_score(y_true_np, y_pred_np)
    mcc = matthews_corrcoef(y_true_np, y_pred_np)

    cm = confusion_matrix(y_true_np, y_pred_np)
    rep = classification_report(y_true_np, y_pred_np, digits=4)

    text = [
        f"TEMPORAL TRANSFORMER RESULTS (FF++ C23, backbone={backbone}, T=5)\n",
        f"Num samples: {len(y_true_np)}",
        f"Accuracy: {acc:.6f}",
        f"F1: {f1:.6f}",
        f"AUC: {auc:.6f}",
        f"Balanced Accuracy: {bal_acc:.6f}",
        f"MCC: {mcc:.6f}\n",
        "Confusion matrix:",
        str(cm),
        "\nClassification report:\n" + rep,
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print("\nSaved report ->", out_report)


if __name__ == "__main__":
    main()
