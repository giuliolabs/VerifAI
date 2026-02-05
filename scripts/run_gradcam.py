"""
Week 19 – Offline Grad-CAM Heatmap Generator (Xception)
======================================================

Generates Grad-CAM heatmaps for a small set of sample videos defined in:
    experiments/week19/explainability/samples.csv

For each sample video:
- Extracts a representative frame (middle frame)
- Runs Grad-CAM on the Xception video model
- Saves:
    <sample_id>_frame.png
    <sample_id>_gradcam.png

------------------------------------------------
DEPENDENCIES
------------------------------------------------
    pip install torch timm opencv-python pandas tqdm pillow numpy

------------------------------------------------
RUN
------------------------------------------------
python scripts/run_gradcam.py ^
  --checkpoint "experiments/results/ffpp_c23_xception_baseline/best_model.pt" ^
  --samples "experiments/week19/explainability/samples.csv" ^
  --outdir "experiments/week19/explainability/heatmaps"

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- Grad-CAM provides qualitative evidence of spatial attention.
- Heatmaps are generated offline for reproducibility (Week 19 requirement).

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import os
import sys

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ml.models.video.xception import build_xception_binary
from ml.explainability.gradcam import GradCAM, find_last_conv_layer


def load_checkpoint_safely(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Loads a checkpoint with flexible key handling (common patterns supported).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Common patterns:
    # - state_dict directly (OrderedDict)
    # - {"state_dict": ...}
    # - {"model_state_dict": ...}
    # - {"model_state": ...}  <-- YOUR CHECKPOINT FORMAT
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            # could already be a state dict-like mapping
            state = ckpt
    else:
        state = ckpt

    # If keys are prefixed (e.g., "module."), strip them
    cleaned = {}
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


def extract_middle_frame(video_path: str) -> Tuple[np.ndarray, str]:
    """
    Extract the middle frame from a video using OpenCV.

    Returns:
        frame_rgb_uint8 (H, W, 3), and a note string describing extraction
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


def preprocess_for_xception(frame_rgb: np.ndarray) -> torch.Tensor:
    """
    Xception expects 299x299. We use ImageNet mean/std (timm default).
    """
    # Resize to 299x299
    img = cv2.resize(frame_rgb, (299, 299), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(img)

    # ImageNet normalization
    arr = np.array(pil).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,3,299,299]
    return tensor


def save_rgb_png(path: str, rgb_uint8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def sigmoid_prob_from_logit(logit: torch.Tensor) -> float:
    return float(torch.sigmoid(logit).detach().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--samples", default="experiments/week19/explainability/samples.csv")
    parser.add_argument("--outdir", default="experiments/week19/explainability/heatmaps")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.samples):
        raise FileNotFoundError(f"Samples CSV not found: {args.samples}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Build model + load checkpoint
    model = build_xception_binary(pretrained=False)
    load_checkpoint_safely(model, args.checkpoint)
    model.eval()
    model.to(device)

    # Choose target layer automatically (last Conv2d)
    target_layer = find_last_conv_layer(model)
    cam = GradCAM(model=model, target_layer=target_layer)

    df = pd.read_csv(args.samples)
    os.makedirs(args.outdir, exist_ok=True)

    for row in tqdm(df.to_dict(orient="records"), desc="Grad-CAM"):
        sample_id = row["sample_id"]
        video_path = str(row["video_path"])

        try:
            frame_rgb, note = extract_middle_frame(video_path)
            input_tensor = preprocess_for_xception(frame_rgb).to(device)

            # Compute gradcam
            result = cam(input_tensor=input_tensor, original_rgb=frame_rgb, target_class=None)

            with torch.no_grad():
                out = model(input_tensor)
                if out.ndim == 2 and out.shape[1] == 1:
                    logit = out[0, 0]
                else:
                    logit = out.squeeze()[0] if out.ndim > 0 else out
                prob_fake = sigmoid_prob_from_logit(logit)

            print(f"[SANITY] {sample_id} logit={float(logit):.6f} prob_fake={prob_fake:.6e} path={video_path}")

            # Save original frame + overlay
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
