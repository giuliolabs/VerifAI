"""
Evaluate MobileNetV2 Baseline on FaceForensics++ C23 (Video-Level)
==================================================================

This script evaluates the trained MobileNetV2 visual baseline on the
FaceForensics++ C23 test split using video-level metrics. Although the
model predicts frame by frame, the final task is video classification,
so frame probabilities are grouped by video and averaged to produce one
score per video.

Run (from project root):
    python ml/evaluate.py

Outputs (in experiments/results/ffpp_c23_mobilenet_baseline):
    - video_level_report.txt
    - video_level_metrics.json

This evaluation uses the shared metrics utilities so results stay
consistent with the rest of the VerifAI framework.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Add project root to Python path so internal modules can be imported correctly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# defaultdict is used to collect many frame probabilities per video
from collections import defaultdict

# PyTorch core utilities
import torch
from torch.utils.data import DataLoader

# NumPy is used for readable confusion matrix printing
import numpy as np

# Project-specific transforms, dataset loader, model builder, and shared metrics utilities
from ml.augmentations import get_transforms
from ml.data_loader import FFPPFrameDataset
from ml.models.video.mobilenet_baseline import build_model
from ml.metrics import (
    aggregate_video_scores_mean,
    compute_binary_metrics,
    save_metrics_report,
)


@torch.no_grad()
def main():
    """
    Main evaluation pipeline for the MobileNetV2 baseline.

    This function:
    - loads the FaceForensics++ C23 test dataset
    - restores the trained MobileNetV2 checkpoint
    - runs frame-level inference
    - aggregates frame probabilities into video-level scores
    - computes consistent binary metrics
    - saves TXT and JSON evaluation outputs
    """
    data_root = Path("data/interim/frames/FaceForensics++_C23")
    out_dir = Path("experiments/results/ffpp_c23_mobilenet_baseline")
    model_path = out_dir / "best_model.pt"

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run ml/train.py first.")

    # Build test dataset.
    # The mapping_file is required because the dataset uses hashed SAFE_ID folders.
    test_dir = data_root / "test"
    test_ds = FFPPFrameDataset(
        str(test_dir),
        transform=get_transforms(train=False),
        mapping_file=r"data\interim\frames\FaceForensics++_C23\test\_id_map.csv"
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    # Rebuild model architecture and load trained weights
    model = build_model().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Store frame probabilities grouped by video.
    # These will later be averaged into one probability per video.
    video_probs = defaultdict(list)
    video_label = {}

    for images, labels, video_ids in test_loader:
        images = images.to(device)

        # Forward pass on individual frames
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

        # Group frame probabilities by video_id
        for prob, label, video_id in zip(probs, labels, video_ids):
            video_probs[video_id].append(float(prob))

            # Save label only once per video
            if video_id not in video_label:
                video_label[video_id] = int(label)

    # Convert frame-level probabilities into video-level scores
    # by taking the mean probability for each video
    y_true, y_score, video_ids = aggregate_video_scores_mean(video_probs, video_label)

    # Compute framework-consistent binary metrics
    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        include_curves=True,
    )

    print("\nVIDEO-LEVEL RESULTS (FF++ C23 MobileNetV2 baseline)")
    print("Num videos:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)

    # Save both human-readable TXT and machine-readable JSON outputs
    txt_path, json_path = save_metrics_report(
        out_dir=out_dir,
        name="video_level",
        metrics=metrics,
        extra={
            "dataset": "FaceForensics++ C23",
            "aggregation": "mean(frame_probs)",
            "model_path": str(model_path),
        },
    )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()