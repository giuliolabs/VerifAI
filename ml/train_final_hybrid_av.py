"""
Train Final Hybrid AV Fusion for VerifAI
========================================

This script trains the final late-fusion audio-visual classifier used
in VerifAI. Instead of training a single end-to-end multimodal network,
it combines the outputs of two already trained specialist backbones:

- a final visual backbone trained across all available frame datasets
- a final audio backbone trained on FakeAVCeleb MFCC features

The fusion model is then learned on FakeAVCeleb and the final decision
threshold is tuned on the validation split.

Uses:
- visual backbone trained on all frame datasets
- audio backbone trained on FakeAVCeleb
- learns late fusion on FakeAVCeleb train
- tunes threshold on FakeAVCeleb val

Run:
    python -m ml.train_final_hybrid_av

Output:
    experiments/results/final_hybrid_av/fusion_model.joblib
    experiments/results/final_hybrid_av/fusion_meta.json

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path so internal modules can be imported correctly
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Standard library for saving metadata
import json

# joblib is used to save the trained fusion classifier
import joblib

# NumPy is used for feature concatenation and threshold search
import numpy as np

# PyTorch is used for model loading and tensor-based inference
import torch
import torch.nn.functional as f

# tqdm provides progress bars during feature extraction
from tqdm import tqdm

# Logistic regression is used as the final late-fusion classifier
from sklearn.linear_model import LogisticRegression

# Balanced accuracy is used to tune the decision threshold fairly across classes
from sklearn.metrics import balanced_accuracy_score

# DataLoader batches the multimodal dataset
from torch.utils.data import DataLoader

# Project dataset and model builders
from ml.av_data_loader import FakeAVCelebAVDataset
from ml.models.video.xception import build_xception_binary
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary


# Paths to pretrained specialist backbones
VISUAL_CKPT = Path("experiments/results/final_visual_all_datasets_xception/best_model.pt")
AUDIO_CKPT = Path("experiments/results/fakeavceleb_audio_resnet_baseline/best_model.pt")

# Output directory for the final fusion artifacts
OUT_DIR = Path("experiments/results/final_hybrid_av")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _torch_load_compat(path: Path, device: str):
    """
    Load a PyTorch checkpoint in a version-compatible way.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def extract_features(split: str, visual_model, audio_model, device: str):
    """
    Extract late-fusion features for one dataset split.

    For each sample, this function computes:
    - visual fake probability
    - audio fake probability

    These two probabilities are then stacked into a 2D feature vector
    used later by the logistic regression fusion model.
    """
    ds = FakeAVCelebAVDataset(split, strict=True, mfcc_max_len=200)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    x_all = []
    y_all = []

    for video, mfcc, y, _ in tqdm(loader, desc=f"Extract {split}"):
        video = video.to(device)   # [B, T, 3, 224, 224]
        mfcc = mfcc.to(device)

        # Flatten temporal dimension so each frame can be processed
        # by the frame-level visual backbone
        batch_size, time_steps, channels, height, width = video.shape
        frames = video.view(batch_size * time_steps, channels, height, width)

        # Resize frames to 299x299 because Xception expects this resolution
        frames = f.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)

        # Visual branch:
        # compute per-frame logits, then average probabilities across time
        visual_logits = visual_model(frames).view(batch_size, time_steps)
        visual_prob = torch.sigmoid(visual_logits).mean(dim=1)

        # Audio branch:
        # compute one probability per MFCC tensor
        audio_logits = audio_model(mfcc).squeeze(1)
        audio_prob = torch.sigmoid(audio_logits)

        # Final fusion input is a simple 2-feature vector:
        # [visual_prob, audio_prob]
        features = torch.stack([visual_prob, audio_prob], dim=1).cpu().numpy()

        x_all.append(features)
        y_all.append(y.cpu().numpy().astype(int))

    # Merge all mini-batches into full arrays
    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all


def main():
    """
    Main training pipeline for the final hybrid AV fusion model.

    This function:
    - loads the pretrained visual and audio backbones
    - extracts probabilities from both branches
    - trains a logistic regression late-fusion classifier
    - tunes the decision threshold on validation data
    - saves the fusion model and metadata
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Validate that both required backbone checkpoints exist
    if not VISUAL_CKPT.exists():
        raise FileNotFoundError(f"Missing visual checkpoint: {VISUAL_CKPT}")
    if not AUDIO_CKPT.exists():
        raise FileNotFoundError(f"Missing audio checkpoint: {AUDIO_CKPT}")

    # Rebuild model architectures without pretrained ImageNet weights,
    # because learned project checkpoints will be loaded immediately after
    visual_model = build_xception_binary(pretrained=False).to(device)
    audio_model = build_mfcc_resnet18_binary(pretrained=False).to(device)

    # Load saved trained weights
    visual_ckpt = _torch_load_compat(VISUAL_CKPT, device)
    audio_ckpt = _torch_load_compat(AUDIO_CKPT, device)

    visual_model.load_state_dict(visual_ckpt["model_state"])
    audio_model.load_state_dict(audio_ckpt["model_state"])

    # Switch both models to evaluation mode for stable inference
    visual_model.eval()
    audio_model.eval()

    # Extract fusion training features from FakeAVCeleb train split
    x_train, y_train = extract_features("train", visual_model, audio_model, device)

    # Extract validation features used for threshold tuning
    x_val, y_val = extract_features("val", visual_model, audio_model, device)

    # Train a simple balanced logistic regression model on the two modality probabilities.
    # This is a clean late-fusion strategy that is easy to interpret and deploy.
    fusion = LogisticRegression(max_iter=1000, class_weight="balanced")
    fusion.fit(x_train, y_train)

    # Compute validation probabilities for threshold tuning
    val_prob = fusion.predict_proba(x_val)[:, 1]

    # Search for the threshold that maximizes balanced accuracy.
    # Balanced accuracy is used because it is less biased by class imbalance.
    best_t = 0.5
    best_bal = -1.0

    for t in np.linspace(0.05, 0.95, 19):
        preds = (val_prob >= t).astype(int)
        bal = balanced_accuracy_score(y_val, preds)

        if bal > best_bal:
            best_bal = float(bal)
            best_t = float(t)

    # Save the trained fusion classifier
    joblib.dump(fusion, OUT_DIR / "fusion_model.joblib")

    # Save metadata so the deployment layer knows:
    # - which backbones were used
    # - which threshold was selected
    # - what the fusion input features represent
    meta = {
        "visual_checkpoint": str(VISUAL_CKPT),
        "audio_checkpoint": str(AUDIO_CKPT),
        "best_threshold": best_t,
        "best_val_balanced_accuracy": best_bal,
        "features": ["visual_prob", "audio_prob"],
        "dataset": "FakeAVCeleb",
        "model": "Final hybrid AV fusion",
    }
    (OUT_DIR / "fusion_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )

    print("Saved fusion model ->", OUT_DIR / "fusion_model.joblib")
    print("Saved metadata ->", OUT_DIR / "fusion_meta.json")
    print("Best threshold:", best_t)
    print("Best val balanced accuracy:", best_bal)


if __name__ == "__main__":
    main()