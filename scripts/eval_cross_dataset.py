"""
Cross-Dataset Generalization Evaluation (Week 19)
================================================

Evaluates how a deepfake detection model trained on one dataset
generalizes to an unseen dataset.

This script is designed for *real-only* evaluation datasets
(e.g. Survey369), where all inputs are genuine videos collected
from real users.

The goal is to measure robustness by estimating the
False Positive Rate (FPR): how often real videos are incorrectly
classified as fake.

------------------------------------------------
INPUT
------------------------------------------------
- A CSV manifest with columns:
    video_path,label,video_id

  Example (real-only):
    data/raw/Survey369/videos/21_16_Luigi.mp4,real,S001

------------------------------------------------
MODEL
------------------------------------------------
- Uses an existing trained checkpoint
- Frame-level inference (single representative frame per video)

------------------------------------------------
OUTPUT
------------------------------------------------
Saved to: experiments/cross_dataset/

1) predictions CSV:
   survey369_predictions.csv

2) metrics summary CSV:
   metrics_summary.csv

------------------------------------------------
METRICS REPORTED
------------------------------------------------
- Total videos
- Number predicted as fake
- False Positive Rate (FPR)
- Average fake probability

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
This evaluation simulates real-world deployment where a model
encounters unseen, user-generated content. A low false positive
rate is critical for trust and usability.

Author: Giulio Dajani
Project: VerifAI – Multimodal Deepfake Detection Framework
"""

import os
import sys
import csv
import argparse

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Ensure project root on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.video.xception import build_xception_binary


# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)

    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned, strict=False)


def extract_middle_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Frame extraction failed")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preprocess(frame: np.ndarray) -> torch.Tensor:
    img = cv2.resize(frame, (299, 299))
    arr = img.astype(np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).float()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--csv", required=True, help="Input CSV (real-only)")
    parser.add_argument("--outdir", default="experiments/cross_dataset")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    # Load model
    model = build_xception_binary(pretrained=False)
    load_checkpoint(model, args.checkpoint)
    model = model.float()
    model.to(device)
    model.eval()

    df = pd.read_csv(args.csv)

    rows = []
    fake_count = 0
    probs = []

    for r in tqdm(df.to_dict("records"), desc="Cross-dataset eval"):
        try:
            frame = extract_middle_frame(r["video_path"])
            x = preprocess(frame).to(device)

            with torch.no_grad():
                logit = model(x).squeeze(1)
                prob_fake = torch.sigmoid(logit).item()

            pred = "fake" if prob_fake >= 0.5 else "real"
            if pred == "fake":
                fake_count += 1

            probs.append(prob_fake)

            rows.append([
                r["video_id"],
                r["video_path"],
                prob_fake,
                pred
            ])

        except Exception as e:
            print(f"[FAIL] {r['video_id']}: {e}")

    # Save predictions
    pred_path = os.path.join(args.outdir, "survey369_predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "video_path", "prob_fake", "predicted_label"])
        w.writerows(rows)

    # Metrics
    total = len(rows)
    fpr = fake_count / total if total else 0.0
    avg_prob = float(np.mean(probs)) if probs else 0.0

    metrics_path = os.path.join(args.outdir, "metrics_summary.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["total_videos", total])
        w.writerow(["predicted_fake", fake_count])
        w.writerow(["false_positive_rate", round(fpr, 4)])
        w.writerow(["average_prob_fake", round(avg_prob, 4)])

    print("\nDone.")
    print("Predictions:", pred_path)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()
