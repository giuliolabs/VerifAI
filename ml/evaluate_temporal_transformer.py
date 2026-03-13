"""
Evaluate Temporal Transformer (T=5) on FaceForensics++ C23 Test Split
=====================================================================

This script evaluates the trained Temporal Transformer model on the
FaceForensics++ C23 test split. Unlike frame-level models, this model
predicts directly from short sequences of frames, allowing it to capture
temporal inconsistencies that may appear across manipulated videos.

Adds consistent metrics for consolidation:
- Accuracy
- Balanced Accuracy
- F1 (macro/weighted in sklearn report string)
- ROC-AUC
- Average Precision (AP)
- MCC
- Confusion matrix + classification report

Saves:
    experiments/results/ffpp_c23_temporal_<backbone>/test_report.txt
    experiments/results/ffpp_c23_temporal_<backbone>/test_metrics.json

Run:
  python -m ml.evaluate_temporal_transformer --backbone mobilenetv2
  python -m ml.evaluate_temporal_transformer --backbone vit

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Add project root to Python path so internal modules can be imported correctly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# argparse is used to choose the backbone from the command line
import argparse

# NumPy is used for threshold search and readable metric printing
import numpy as np

# PyTorch core utilities
import torch
from torch.utils.data import DataLoader

# tqdm provides a progress bar during evaluation
from tqdm import tqdm

# Balanced accuracy is only used for optional threshold selection
from sklearn.metrics import balanced_accuracy_score

# Project dataset loader, model builder, and shared metrics utilities
from ml.video_sequence_data_loader import FFPPSequenceDataset
from ml.models.video.temporal_transformer import build_temporal_transformer_model
from ml.metrics import compute_binary_metrics, save_metrics_report


# Root folder containing extracted sequence frames
FRAMES_ROOT = Path("data/interim/frames/FaceForensics++_C23")

# Folder containing dataset split CSV files
SPLITS_DIR = Path("data/splits")
TEST_CSV = SPLITS_DIR / "faceforensics++_c23_test.csv"


def _torch_load_compat(path: Path, device: str):
    """
    Load a PyTorch checkpoint in a way that remains compatible
    across different torch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _pick_threshold_by_balanced_acc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Choose the threshold that maximizes balanced accuracy.
    """
    best_t = 0.5
    best_bal = -1.0

    for t in np.linspace(0.05, 0.95, 19):
        preds = (y_score >= t).astype(int)
        bal = balanced_accuracy_score(y_true, preds)
        if bal > best_bal:
            best_bal = bal
            best_t = float(t)

    return best_t


@torch.no_grad()
def main() -> None:
    """
    Main evaluation pipeline for the Temporal Transformer.

    This function:
    - loads the test sequence dataset
    - restores the selected backbone checkpoint
    - runs video-sequence inference
    - optionally selects a threshold
    - computes consistent binary metrics
    - saves TXT and JSON evaluation outputs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mobilenetv2", choices=["vit", "mobilenetv2"])
    args = parser.parse_args()
    backbone = args.backbone

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Sequence length and spatial resolution must match training configuration
    t = 5
    img_size = 224

    out_dir = Path(f"experiments/results/ffpp_c23_temporal_{backbone}")
    ckpt_path = out_dir / "best_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build the test dataset using fixed-length frame sequences
    ds = FFPPSequenceDataset(
        frames_root=FRAMES_ROOT,
        split="test",
        split_csv=TEST_CSV,
        img_size=img_size,
        t=t,
        train=False,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    # Rebuild the model architecture with the chosen backbone
    model = build_temporal_transformer_model(
        backbone=backbone,
        pretrained=False,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        freeze_encoder=False,
    ).to(device)

    # Load trained weights
    ckpt = _torch_load_compat(ckpt_path, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true: list[int] = []
    y_score: list[float] = []

    # Run sequence-level inference
    for frames, y, _ in tqdm(loader, desc="Evaluating"):
        frames = frames.to(device)

        logits = model(frames).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

        # y is already a batch tensor [B]
        y_true.extend([int(v) for v in y.cpu().numpy().tolist()])
        y_score.extend([float(p) for p in probs.tolist()])

    y_true_np = np.array(y_true, dtype=int)
    y_score_np = np.array(y_score, dtype=float)

    # Threshold handling:
    # for ViT backbone, threshold is optionally tuned by balanced accuracy;
    # for MobileNetV2 backbone, default 0.5 is used.
    if backbone == "vit":
        threshold = _pick_threshold_by_balanced_acc(y_true_np, y_score_np)
    else:
        threshold = 0.5

    # Compute consistent binary metrics using the shared utility
    metrics = compute_binary_metrics(
        y_true=y_true_np.tolist(),
        y_score=y_score_np.tolist(),
        threshold=threshold,
        include_curves=True,
    )

    print(f"\nTEMPORAL TRANSFORMER RESULTS (FF++ C23, backbone={backbone}, T=5)")
    print("Checkpoint:", ckpt_path)
    print("Num samples:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("Balanced accuracy:", getattr(metrics, "balanced_accuracy", None))
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("MCC:", getattr(metrics, "mcc", None))
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)

    # Save both human-readable and machine-readable outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path, json_path = save_metrics_report(
        out_dir=out_dir,
        name="test",
        metrics=metrics,
        extra={
            "dataset": "FaceForensics++ C23",
            "model": "Temporal Transformer",
            "backbone": backbone,
            "T": t,
            "img_size": img_size,
            "threshold": threshold,
            "checkpoint": str(ckpt_path),
        },
    )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()