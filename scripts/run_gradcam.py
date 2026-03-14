"""
Week 19 – Offline Grad-CAM Heatmap Generator (Xception)
======================================================

This script generates Grad-CAM heatmaps for a set of sample videos using
the trained Xception visual deepfake detection model.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which
regions of an image influenced the model's prediction. In this project,
it is used to demonstrate the spatial attention patterns of the visual
deepfake detector.

Heatmaps are generated offline for reproducibility, as required for
Week 19 explainability analysis.

For each sample video:
- The script extracts a representative frame (the middle frame)
- Runs Grad-CAM using the Xception model
- Saves:
      <sample_id>_frame.png
      <sample_id>_gradcam.png

Input samples are defined in:

    experiments/week19/explainability/samples.csv

------------------------------------------------
DEPENDENCIES
------------------------------------------------
Install required packages:

    pip install torch timm opencv-python pandas tqdm pillow numpy

------------------------------------------------
RUN
------------------------------------------------
Example command (from project root):

python scripts/run_gradcam.py ^
  --checkpoint "experiments/results/ffpp_c23_xception_baseline/best_model.pt" ^
  --samples "experiments/week19/explainability/samples.csv" ^
  --outdir "experiments/week19/explainability/heatmaps"

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- Grad-CAM provides qualitative explainability evidence.
- The heatmaps show which spatial regions influence predictions.
- All heatmaps are generated offline for reproducibility.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

import os
import sys

# -------------------------------------------------------
# Ensure project root is on the Python path
# -------------------------------------------------------
# This allows imports like:
#     from ml.models.video.xception import build_xception_binary
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------
# Standard library imports
# -------------------------------------------------------
import argparse
from typing import Tuple

# -------------------------------------------------------
# Third-party imports
# -------------------------------------------------------
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------
# Project imports
# -------------------------------------------------------
from ml.models.video.xception import build_xception_binary
from ml.explainability.gradcam import GradCAM, find_last_conv_layer


# -------------------------------------------------------
# Checkpoint loader with flexible format support
# -------------------------------------------------------
def load_checkpoint_safely(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load a model checkpoint while handling multiple common formats.

    Checkpoints may store weights in different structures depending on
    how training scripts saved them. This function attempts to support:

        state_dict directly
        {"state_dict": ...}
        {"model_state_dict": ...}
        {"model_state": ...}

    It also removes prefixes like "module." used when models were trained
    using DataParallel.
    """

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):

        if "model_state" in ckpt:
            state = ckpt["model_state"]

        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]

        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]

        else:
            state = ckpt

    else:
        state = ckpt

    cleaned = {}

    # Remove DataParallel prefixes if present
    for key, value in state.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        cleaned[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    if missing:
        print(f"[WARN] Missing keys (partial load): {len(missing)}")

    if unexpected:
        print(f"[WARN] Unexpected keys ignored: {len(unexpected)}")


# -------------------------------------------------------
# Frame extraction from video
# -------------------------------------------------------
def extract_middle_frame(video_path: str) -> Tuple[np.ndarray, str]:
    """
    Extract the middle frame from a video file using OpenCV.

    Returns:
        frame_rgb_uint8 : numpy array (H, W, 3)
        note            : text describing which frame was extracted
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("OpenCV could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid video or zero frames.")

    target_index = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)

    ok, frame_bgr = cap.read()
    cap.release()

    if not ok or frame_bgr is None:
        raise RuntimeError("Failed to read middle frame.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return frame_rgb, f"middle_frame={target_index}/{total_frames}"


# -------------------------------------------------------
# Preprocessing for Xception model
# -------------------------------------------------------
def preprocess_for_xception(frame_rgb: np.ndarray) -> torch.Tensor:
    """
    Prepare an image for Xception inference.

    Xception expects:
        - image size: 299x299
        - ImageNet normalization
    """

    img = cv2.resize(frame_rgb, (299, 299), interpolation=cv2.INTER_AREA)

    pil = Image.fromarray(img)

    arr = np.array(pil).astype(np.float32) / 255.0

    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )

    arr = np.transpose(arr, (2, 0, 1))

    tensor = torch.from_numpy(arr).unsqueeze(0)

    return tensor


# -------------------------------------------------------
# Save RGB image as PNG
# -------------------------------------------------------
def save_rgb_png(path: str, rgb_uint8: np.ndarray) -> None:
    """
    Save an RGB image to disk using OpenCV.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, bgr)


# -------------------------------------------------------
# Convert model logit into probability
# -------------------------------------------------------
def sigmoid_prob_from_logit(logit: torch.Tensor) -> float:
    """
    Convert a raw model logit to probability using sigmoid.
    """
    return float(torch.sigmoid(logit).detach().cpu().item())


# -------------------------------------------------------
# Main execution
# -------------------------------------------------------
def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint (.pt)"
    )

    parser.add_argument(
        "--samples",
        default="experiments/week19/explainability/samples.csv"
    )

    parser.add_argument(
        "--outdir",
        default="experiments/week19/explainability/heatmaps"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"]
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # Validate input files
    # -------------------------------------------------------
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if not os.path.exists(args.samples):
        raise FileNotFoundError(f"Samples CSV not found: {args.samples}")

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    # -------------------------------------------------------
    # Load model
    # -------------------------------------------------------
    model = build_xception_binary(pretrained=False)

    load_checkpoint_safely(model, args.checkpoint)

    model.eval()
    model.to(device)

    # Automatically locate last convolution layer for Grad-CAM
    target_layer = find_last_conv_layer(model)

    cam = GradCAM(model=model, target_layer=target_layer)

    # -------------------------------------------------------
    # Load samples
    # -------------------------------------------------------
    df = pd.read_csv(args.samples)

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------------------------------------
    # Process each sample video
    # -------------------------------------------------------
    for row in tqdm(df.to_dict(orient="records"), desc="Grad-CAM"):

        sample_id = row["sample_id"]
        video_path = str(row["video_path"])

        try:

            frame_rgb, note = extract_middle_frame(video_path)

            input_tensor = preprocess_for_xception(frame_rgb).to(device)

            result = cam(
                input_tensor=input_tensor,
                original_rgb=frame_rgb,
                target_class=None
            )

            with torch.no_grad():

                out = model(input_tensor)

                if out.ndim == 2 and out.shape[1] == 1:
                    logit = out[0, 0]

                else:
                    logit = out.squeeze()[0] if out.ndim > 0 else out

                prob_fake = sigmoid_prob_from_logit(logit)

            print(
                f"[SANITY] {sample_id} logit={float(logit):.6f} "
                f"prob_fake={prob_fake:.6e} path={video_path}"
            )

            frame_out = os.path.join(args.outdir, f"{sample_id}_frame.png")
            cam_out = os.path.join(args.outdir, f"{sample_id}_gradcam.png")

            save_rgb_png(frame_out, frame_rgb)
            save_rgb_png(cam_out, result.overlay_rgb)

        except Exception as e:

            print(f"[FAIL] {sample_id}: {video_path} -> {type(e).__name__}: {e}")

    cam.close()

    print(f"Done. Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()