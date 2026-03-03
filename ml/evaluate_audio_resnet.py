"""
Evaluate Audio-Only MFCC->ResNet model (FakeAVCeleb).

Outputs (in experiments/results/fakeavceleb_audio_resnet_baseline):
- test_report.txt
- test_metrics.json

Run:
    python -m ml.evaluate_audio_resnet_baseline
(or if you call it as a script:)
    python ml/evaluate_audio_resnet_baseline.py

Dependencies:
    pip install torch torchvision
    pip install numpy scikit-learn tqdm
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.audio_data_loader import MFCCDataset
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary
from ml.metrics import compute_binary_metrics, save_metrics_report


@torch.no_grad()
def main():
    # --- data ---
    test_csv = "data/splits/fakeavceleb_test.csv"
    mfcc_test_root = "data/processed/audio_features/FakeAVCeleb_v1.2/test"

    # --- paths ---
    ckpt_path = Path("experiments/results/fakeavceleb_audio_resnet_baseline/best_model.pt")
    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # --- loader ---
    ds = MFCCDataset(test_csv, mfcc_test_root)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    # --- model ---
    model = build_mfcc_resnet18_binary(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- collect probs + labels ---
    y_true: list[int] = []
    y_score: list[float] = []

    for x, y, _ in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        logits = model(x).squeeze(1)                 # [B]
        probs = torch.sigmoid(logits).cpu().numpy()  # [B] probabilities

        # y is usually a tensor on CPU from the dataset
        y_true.extend([int(v) for v in y.numpy().tolist()])
        y_score.extend([float(p) for p in probs.tolist()])

    # --- metrics (single source of truth) ---
    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        include_curves=True,
    )

    print("\nAUDIO-ONLY RESULTS (FakeAVCeleb MFCC->ResNet18 baseline)")
    print("Num samples:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("Balanced accuracy:", getattr(metrics, "balanced_accuracy", None))
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("MCC:", getattr(metrics, "mcc", None))
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)

    txt_path, json_path = save_metrics_report(
        out_dir=out_dir,
        name="test",
        metrics=metrics,
        extra={
            "dataset": "FakeAVCeleb",
            "input": "MFCC",
            "model": "ResNet18",
            "checkpoint": str(ckpt_path),
        },
    )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()
