"""
Evaluate Visual CNN (Xception) on FaceForensics++ C23 (Video-Level Metrics)
===========================================================================

This script evaluates the trained Xception visual baseline on the
FaceForensics++ C23 test split using video-level metrics rather than
frame-level metrics. This is important because the final task is video
classification, so frame predictions are aggregated into a single score
per video before evaluation.

What it does
------------
- Loads FF++ C23 test frames (hashed folders) via FFPPFrameDataset
- Loads checkpoint: experiments/results/ffpp_c23_xception_baseline/best_model.pt
- Runs frame inference -> aggregates to video score by mean(frame_prob)
- Computes and saves consistent metrics for consolidation:
    - Accuracy
    - Balanced Accuracy
    - F1 (weighted + macro via sklearn report inside compute_binary_metrics)
    - ROC-AUC
    - Average Precision (AP)
    - MCC
    - Confusion matrix + classification report

Saves:
    experiments/results/ffpp_c23_xception_baseline/test_report.txt
    experiments/results/ffpp_c23_xception_baseline/test_metrics.json

Run:
    python -m ml.evaluate_xception

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Add project root to Python path so internal modules can be imported
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# defaultdict is used to collect multiple frame probabilities per video
from collections import defaultdict

# NumPy is used for readable matrix printing
import numpy as np

# PyTorch core utilities
import torch
from torch.utils.data import DataLoader

# torchvision transforms prepare frames for Xception input
from torchvision import transforms

# tqdm provides progress bars during evaluation
from tqdm import tqdm

# Project dataset loader, model builder, and shared metrics utilities
from ml.data_loader import FFPPFrameDataset
from ml.models.video.xception import build_xception_binary
from ml.metrics import (
    aggregate_video_scores_mean,
    compute_binary_metrics,
    save_metrics_report,
)


def make_transforms():
    """
    Build deterministic preprocessing pipeline for test-time evaluation.

    Xception expects 299x299 input resolution and ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _torch_load_compat(path: Path, device: str):
    """
    Load PyTorch checkpoint in a way that remains compatible across
    different torch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def main():
    """
    Main evaluation pipeline for the Xception visual baseline.

    This function:
    - loads the test dataset and best checkpoint
    - runs frame-level inference
    - aggregates frame probabilities into video-level scores
    - computes consistent binary metrics
    - saves TXT and JSON reports
    """
    data_root = Path("data/interim/frames/FaceForensics++_C23")
    test_dir = data_root / "test"

    ckpt_path = Path("experiments/results/ffpp_c23_xception_baseline/best_model.pt")
    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # IMPORTANT:
    # FFPPFrameDataset needs the mapping file when using hashed SAFE_ID folders,
    # so frames can still be linked back to the correct video identities.
    ds = FFPPFrameDataset(
        str(test_dir),
        transform=make_transforms(),
        mapping_file=r"data\interim\frames\FaceForensics++_C23\test\_id_map.csv",
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    # Rebuild Xception architecture and load the trained checkpoint
    model = build_xception_binary(pretrained=False).to(device)
    ckpt = _torch_load_compat(ckpt_path, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Store frame-level probabilities grouped by video_id.
    # These will later be averaged into one video-level score.
    video_probs = defaultdict(list)
    video_label = {}

    for images, labels, video_ids in tqdm(loader, desc="Evaluating"):
        images = images.to(device)

        # Forward pass on individual frames
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        # Group frame probabilities by video
        for p, y, vid in zip(probs, labels, video_ids):
            vid = str(vid)
            y = int(y)

            video_probs[vid].append(float(p))

            # Save label once per video
            if vid not in video_label:
                video_label[vid] = y

    # Aggregate frame-level probabilities into one score per video
    # using the shared mean-probability helper for consistency.
    y_true, y_score, vids = aggregate_video_scores_mean(video_probs, video_label)

    # Compute binary classification metrics at video level
    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        include_curves=True,  # also stores ROC/PR curve data when possible
    )

    print("\nVIDEO-LEVEL RESULTS (FF++ C23 Xception baseline)")
    print("Num videos:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)

    # Save both human-readable TXT and machine-readable JSON outputs
    txt_path, json_path = save_metrics_report(
        out_dir=out_dir,
        name="test",
        metrics=metrics,
        extra={
            "dataset": "FaceForensics++ C23",
            "aggregation": "mean(frame_probs)",
            "checkpoint": str(ckpt_path),
            "split": "test",
            "model": "Xception baseline",
        },
    )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()