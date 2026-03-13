"""
Evaluate Audio-Only MFCC -> ResNet Model (FakeAVCeleb)
======================================================

This script evaluates the trained MFCC-based audio baseline on the
FakeAVCeleb test split. It loads the saved ResNet18 checkpoint,
runs inference on precomputed MFCC features, and produces a full
binary classification summary for reporting and comparison.

Outputs:
- Accuracy, F1 (binary), ROC-AUC, Average Precision (AP)
- Balanced Accuracy, MCC
- Confusion matrix
- Classification report

Saved report:
    experiments/results/fakeavceleb_audio_resnet_baseline/test_report.txt

Saved metrics JSON:
    experiments/results/fakeavceleb_audio_resnet_baseline/test_metrics.json

Run:
    python -m ml.evaluate_audio_resnet

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Add project root to Python path so internal modules can be imported correctly
import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# NumPy is used for array conversion and safe NaN handling
import numpy as np

# PyTorch is used for model loading and inference
import torch
from torch.utils.data import DataLoader

# tqdm provides a progress bar during evaluation
from tqdm import tqdm

# sklearn metrics are used to compute the evaluation summary
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
)

# Project dataset loader and MFCC ResNet model builder
from ml.audio_data_loader import MFCCDataset
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary


def _torch_load_compat(path: Path, device: str):
    """
    Load a PyTorch checkpoint in a version-compatible way.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def main():
    """
    Main evaluation pipeline for the MFCC -> ResNet18 audio baseline.

    This function:
    - loads the FakeAVCeleb test MFCC dataset
    - restores the trained checkpoint
    - runs binary inference
    - computes evaluation metrics
    - saves both human-readable and machine-readable outputs
    """
    test_csv = "data/splits/fakeavceleb_test.csv"
    mfcc_test_root = "data/processed/audio_features/FakeAVCeleb_v1.2/test"

    ckpt_path = Path("experiments/results/fakeavceleb_audio_resnet_baseline/best_model.pt")
    out_dir = ckpt_path.parent
    out_report = out_dir / "test_report.txt"
    out_json = out_dir / "test_metrics.json"

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build test dataset and loader
    ds = MFCCDataset(test_csv, mfcc_test_root)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    # Rebuild model architecture and load trained weights
    model = build_mfcc_resnet18_binary(pretrained=False).to(device)
    ckpt = _torch_load_compat(ckpt_path, device)

    # Some checkpoints store weights under "model_state",
    # while others may already be raw state dictionaries
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    y_true: list[int] = []
    y_prob: list[float] = []

    # Run inference across the full test set
    for x, y, _ in tqdm(loader, desc="Evaluating"):
        x = x.to(device)

        # Model outputs a single logit per sample
        logits = model(x).squeeze(1)

        # Convert logits into fake probabilities
        probs = torch.sigmoid(logits).cpu().numpy()

        y_true.extend(y.numpy().astype(int).tolist())
        y_prob.extend(probs.tolist())

    # Convert lists to NumPy arrays for metric calculation
    y_true_np = np.array(y_true, dtype=int)
    y_prob_np = np.array(y_prob, dtype=float)
    y_pred_np = (y_prob_np >= 0.5).astype(int)

    # Core binary classification metrics
    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true_np, y_pred_np)
    mcc = matthews_corrcoef(y_true_np, y_pred_np)

    # ROC-AUC and AP require both classes to be present
    try:
        auc = roc_auc_score(y_true_np, y_prob_np) if len(np.unique(y_true_np)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    try:
        ap = average_precision_score(y_true_np, y_prob_np) if len(np.unique(y_true_np)) > 1 else float("nan")
    except ValueError:
        ap = float("nan")

    # Confusion matrix and detailed classification report
    cm = confusion_matrix(y_true_np, y_pred_np)
    report = classification_report(y_true_np, y_pred_np, digits=4, zero_division=0)

    # Build human-readable report text
    text = [
        "AUDIO-ONLY RESULTS (FakeAVCeleb MFCC -> ResNet18 baseline)\n",
        f"Checkpoint: {ckpt_path}",
        f"Num samples: {len(y_true_np)}",
        "",
        f"Accuracy: {acc:.6f}",
        f"Balanced Accuracy: {bal_acc:.6f}",
        f"F1: {f1:.6f}",
        f"ROC-AUC: {auc:.6f}",
        f"Average precision: {ap:.6f}",
        f"MCC: {mcc:.6f}",
        "",
        "Confusion matrix:",
        str(cm),
        "",
        "Classification report:",
        report,
    ]

    # Save TXT report for examiner-friendly reading
    out_report.write_text("\n".join(text), encoding="utf-8")

    # Build JSON payload for consolidation and later comparison scripts
    metrics_payload = {
        "num_samples": int(len(y_true_np)),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1": float(f1),
        "auc_roc": None if np.isnan(auc) else float(auc),
        "ap": None if np.isnan(ap) else float(ap),
        "mcc": float(mcc),
        "confusion_matrix": cm.tolist(),
        "checkpoint": str(ckpt_path),
        "dataset": "FakeAVCeleb_v1.2",
        "model": "mfcc_resnet18",
        "task": "binary_classification",
        "threshold": 0.5,
    }

    # Save machine-readable JSON metrics
    out_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("\n".join(text))
    print("\nSaved report ->", out_report)
    print("Saved metrics json ->", out_json)


if __name__ == "__main__":
    main()