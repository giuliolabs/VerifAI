"""
Evaluate Audio-Only WAV -> 1D CNN Encoder Model (FakeAVCeleb v1.2)
===================================================================

This script evaluates the raw-waveform audio baseline on the FakeAVCeleb
test split. It loads the trained WAV -> 1D CNN checkpoint, performs
binary inference on each sample, and saves consistent evaluation outputs
for later comparison and consolidation.

Writes consistent metrics for consolidation:
- Accuracy
- Balanced Accuracy
- F1 (from sklearn report)
- ROC-AUC
- Average Precision (AP)
- MCC
- Confusion matrix + classification report

Saves:
    experiments/results/fakeavceleb_wav_encoder_baseline/test_report.txt
    experiments/results/fakeavceleb_wav_encoder_baseline/test_metrics.json
    experiments/results/fakeavceleb_wav_encoder_baseline/predictions.csv

Run:
    python -m ml.evaluate_wav_encoder

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Add project root to Python path so internal modules can be imported correctly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# csv is used to save per-sample prediction outputs
import csv

# NumPy is used for thresholding and readable matrix printing
import numpy as np

# PyTorch core utilities
import torch
from torch.utils.data import DataLoader

# tqdm provides a progress bar during evaluation
from tqdm import tqdm

# Project dataset loader, model builder, and shared metrics utilities
from ml.wav_data_loader import WavDataset
from ml.models.audio.wav_encoder import build_wav_binary_classifier
from ml.metrics import compute_binary_metrics, save_metrics_report


def _torch_load_compat(path: Path, device: str):
    """
    Load a PyTorch checkpoint in a way that remains compatible
    across different torch versions.

    Newer PyTorch versions may support weights_only=False explicitly,
    while older versions may not.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def main() -> None:
    """
    Main evaluation pipeline for the WAV -> 1D CNN audio baseline.

    This function:
    - loads the FakeAVCeleb test set
    - restores the trained checkpoint
    - runs inference to obtain fake probabilities
    - computes standard binary metrics
    - saves a TXT report, JSON metrics file, and per-sample predictions CSV
    """
    test_csv = "data/splits/fakeavceleb_test.csv"
    wav_test_root = "data/interim/audio/FakeAVCeleb_v1.2/test"

    ckpt_path = Path("experiments/results/fakeavceleb_wav_encoder_baseline/best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_csv = out_dir / "predictions.csv"

    # Audio settings must match the training configuration
    sample_rate = 16000
    seconds = 3

    # Leave strict sample-rate checking disabled unless all WAV files
    # are guaranteed to match exactly.
    strict_sr = False

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- Data ----
    # Build test dataset using the same waveform settings used during training
    ds = WavDataset(
        split_csv=test_csv,
        wav_root=wav_test_root,
        sample_rate=sample_rate,
        seconds=seconds,
        strict_sr=strict_sr,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    # ---- Model ----
    # Rebuild the model architecture with the same hyperparameters used in training
    model = build_wav_binary_classifier(
        sample_rate=sample_rate,
        embedding_dim=256,
        base_channels=32,
        dropout=0.2,
    ).to(device)

    # Load checkpoint weights
    ckpt = _torch_load_compat(ckpt_path, device)

    # Some checkpoints may store weights inside "model_state",
    # while others may directly store the state dict itself
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    y_true: list[int] = []
    y_score: list[float] = []

    # ---- Inference ----
    for x, y, _ in tqdm(loader, desc="Evaluating"):
        x = x.to(device)              # [B, 1, T]

        # Forward pass returns one logit per sample
        logits = model(x).squeeze(1)  # [B]

        # Convert logits into fake probabilities
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        y_true.extend([int(v) for v in y.cpu().numpy().tolist()])
        y_score.extend([float(p) for p in probs.tolist()])

    # ---- Metrics ----
    # Use the shared metrics utility so evaluation stays consistent
    # across all VerifAI models
    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        include_curves=True,
    )

    # ---- Save per-sample predictions ----
    # Store raw probabilities and thresholded predictions for transparency
    y_pred = (np.array(y_score) >= 0.5).astype(int).tolist()
    with preds_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "prob_fake", "pred"])
        for i, (yt, yp, yhat) in enumerate(zip(y_true, y_score, y_pred)):
            writer.writerow([i, yt, f"{yp:.6f}", yhat])

    # ---- Print summary ----
    print("\nAUDIO-ONLY RESULTS (FakeAVCeleb WAV -> 1D CNN baseline)")
    print("Checkpoint:", ckpt_path)
    print("Num samples:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("Balanced accuracy:", getattr(metrics, "balanced_accuracy", None))
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("MCC:", getattr(metrics, "mcc", None))
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)
    print("\nPredictions CSV:", preds_csv)

    # ---- Save structured report outputs ----
    txt_path, json_path = save_metrics_report(
        out_dir=out_dir,
        name="test",
        metrics=metrics,
        extra={
            "dataset": "FakeAVCeleb v1.2",
            "model": "WAV -> 1D CNN baseline",
            "sample_rate": sample_rate,
            "seconds": seconds,
            "threshold": 0.5,
            "checkpoint": str(ckpt_path),
            "predictions_csv": str(preds_csv),
        },
    )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()