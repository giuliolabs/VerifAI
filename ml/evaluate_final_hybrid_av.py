"""
Evaluate FINAL Hybrid AV Model for VerifAI on FakeAVCeleb Test
==============================================================

This script evaluates the final hybrid audio-visual model used in VerifAI
on the FakeAVCeleb test split. The evaluation combines:

- the final visual backbone trained across all available frame datasets
- the final audio backbone trained on FakeAVCeleb MFCC features
- the learned late-fusion logistic regression classifier

The script extracts visual and audio probabilities for each test sample,
applies the trained fusion model, and computes final binary classification
metrics on the test set.

Run:
    python -m ml.evaluate_final_hybrid_av

Outputs:
    experiments/results/final_hybrid_av/test_report.txt
    experiments/results/final_hybrid_av/test_metrics.json

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Add project root to Python path so internal modules can be imported correctly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# JSON is used to read and write metadata/results files
import json

# joblib is used to load the trained logistic regression fusion model
import joblib

# NumPy is used for array handling and metric preparation
import numpy as np

# PyTorch is used for model loading and inference
import torch
import torch.nn.functional as f

# tqdm provides progress bars during feature extraction
from tqdm import tqdm

# DataLoader batches the multimodal dataset
from torch.utils.data import DataLoader

# sklearn metrics provide the final evaluation summary
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    f1_score,
)

# Project dataset and backbone builders
from ml.av_data_loader import FakeAVCelebAVDataset
from ml.models.video.xception import build_xception_binary
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary


# Paths to trained backbone checkpoints and fusion artifacts
VISUAL_CKPT = Path("experiments/results/final_visual_all_datasets_xception/best_model.pt")
AUDIO_CKPT = Path("experiments/results/fakeavceleb_audio_resnet_baseline/best_model.pt")
FUSION_MODEL_PATH = Path("experiments/results/final_hybrid_av/fusion_model.joblib")
FUSION_META_PATH = Path("experiments/results/final_hybrid_av/fusion_meta.json")
OUT_DIR = Path("experiments/results/final_hybrid_av")


def _torch_load_compat(path: Path, device: str):
    """
    Load a PyTorch checkpoint in a version-compatible way.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def extract_test_features(visual_model, audio_model, device: str):
    """
    Extract late-fusion features from the FakeAVCeleb test split.

    For each sample, this function computes:
    - visual fake probability
    - audio fake probability

    These two values are stacked into a 2D feature vector:
        [visual_prob, audio_prob]

    Returns:
        x_all: feature matrix for fusion classifier
        y_all: ground-truth labels
    """
    ds = FakeAVCelebAVDataset("test", strict=True, mfcc_max_len=200)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    x_all = []
    y_all = []

    for video, mfcc, y, _ in tqdm(loader, desc="Evaluating"):
        video = video.to(device)
        mfcc = mfcc.to(device)

        # Flatten temporal dimension so each frame can be processed
        # independently by the frame-based Xception backbone
        batch_size, time_steps, channels, height, width = video.shape
        frames = video.view(batch_size * time_steps, channels, height, width)

        # Resize frames to 299x299 because Xception expects this resolution
        frames = f.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)

        # Visual branch:
        # produce one probability per frame, then average across time
        visual_logits = visual_model(frames).view(batch_size, time_steps)
        visual_prob = torch.sigmoid(visual_logits).mean(dim=1)

        # Audio branch:
        # produce one probability per MFCC tensor
        audio_logits = audio_model(mfcc).squeeze(1)
        audio_prob = torch.sigmoid(audio_logits)

        # Final fusion input is [visual_prob, audio_prob]
        features = torch.stack([visual_prob, audio_prob], dim=1).cpu().numpy()

        x_all.append(features)
        y_all.append(y.cpu().numpy().astype(int))

    # Merge all mini-batches into complete arrays
    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all


def main():
    """
    Main evaluation pipeline for the final hybrid AV model.

    This function:
    - validates required checkpoints and fusion files
    - loads both trained backbones
    - loads the fusion classifier and saved threshold
    - extracts test features
    - computes final test predictions
    - calculates classification metrics
    - saves results as TXT and JSON
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Validate that all required artifacts exist before starting evaluation
    if not VISUAL_CKPT.exists():
        raise FileNotFoundError(f"Missing visual checkpoint: {VISUAL_CKPT}")
    if not AUDIO_CKPT.exists():
        raise FileNotFoundError(f"Missing audio checkpoint: {AUDIO_CKPT}")
    if not FUSION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing fusion model: {FUSION_MODEL_PATH}")
    if not FUSION_META_PATH.exists():
        raise FileNotFoundError(f"Missing fusion metadata: {FUSION_META_PATH}")

    # Rebuild model architectures without ImageNet pretrained weights,
    # because trained project checkpoints will be loaded immediately after
    visual_model = build_xception_binary(pretrained=False).to(device)
    audio_model = build_mfcc_resnet18_binary(pretrained=False).to(device)

    # Load trained weights for both backbones
    visual_ckpt = _torch_load_compat(VISUAL_CKPT, device)
    audio_ckpt = _torch_load_compat(AUDIO_CKPT, device)

    visual_model.load_state_dict(visual_ckpt["model_state"])
    audio_model.load_state_dict(audio_ckpt["model_state"])

    # Switch models to evaluation mode for stable inference
    visual_model.eval()
    audio_model.eval()

    # Load trained late-fusion classifier and saved best threshold
    fusion = joblib.load(FUSION_MODEL_PATH)
    meta = json.loads(FUSION_META_PATH.read_text(encoding="utf-8"))
    threshold = float(meta["best_threshold"])

    # Extract test features and labels
    x_test, y_true = extract_test_features(visual_model, audio_model, device)

    # Predict fake probabilities using the trained logistic regression fusion model
    y_score = fusion.predict_proba(x_test)[:, 1]

    # Convert probabilities to binary predictions using the tuned threshold
    y_pred = (y_score >= threshold).astype(int)

    # Compute final evaluation metrics
    metrics = {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
        "threshold": threshold,
        "dataset": "FakeAVCeleb",
        "model": "Final hybrid AV (visual-all-datasets + MFCC audio + logistic fusion)",
        "visual_checkpoint": str(VISUAL_CKPT),
        "audio_checkpoint": str(AUDIO_CKPT),
    }

    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save machine-readable JSON metrics
    (OUT_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    # Save human-readable report file
    (OUT_DIR / "test_report.txt").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("Accuracy:", metrics["accuracy"])
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print("ROC-AUC:", metrics["auc_roc"])
    print("AP:", metrics["ap"])
    print("MCC:", metrics["mcc"])
    print("Saved report ->", OUT_DIR / "test_report.txt")
    print("Saved metrics json ->", OUT_DIR / "test_metrics.json")


if __name__ == "__main__":
    main()