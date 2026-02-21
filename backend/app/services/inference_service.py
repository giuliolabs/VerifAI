"""
Inference Service – Multimodal Fusion Model (Render-safe)
========================================================

Key changes for hosted deployments (Render/free tier):
- Force CPU mode (no CUDA on Render anyway)
- Limit torch thread usage (reduces memory spikes)
- Use inference_mode() (lower overhead than no_grad)
- Explicitly delete tensors + run gc to reduce peak RAM
- Keep model loaded once at import time

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

import gc
import os
import tempfile

import torch

from ml.models.fusion.multimodal_fusion import MultimodalFusionModel
from backend.app.services.preprocess_service import extract_features_from_video


# ------------------------------------------------
# Runtime safety for small instances
# ------------------------------------------------
# Render CPU instances often OOM when torch uses many threads.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Render doesn't provide GPU. Forcing CPU avoids any accidental GPU init paths.
_DEVICE = "cpu"


# ------------------------------------------------
# Model loading (once)
# ------------------------------------------------
_MODEL = MultimodalFusionModel().to(_DEVICE)

_CHECKPOINT = "experiments/results/fakeavceleb_av_fusion_v1/best_model.pt"
_state = torch.load(_CHECKPOINT, map_location=_DEVICE)
_MODEL.load_state_dict(_state["model_state"])
_MODEL.eval()


def run_inference(upload_file):
    """
    Perform multimodal deepfake inference on a video upload.

    Returns:
        label (str): "real" or "fake"
        prob_fake (float): probability of fake
    """

    # Write upload to disk (streaming-safe)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upload_file.file.read())
        video_path = tmp.name

    video = None
    mfcc = None

    try:
        video, mfcc = extract_features_from_video(video_path)

        # Add batch dim + move to device
        video = video.unsqueeze(0).to(_DEVICE)  # [1, 5, 3, 224, 224]
        mfcc = mfcc.unsqueeze(0).to(_DEVICE)    # [1, 1, 40, T]

        # inference_mode is faster + lower memory than no_grad
        with torch.inference_mode():
            logit = _MODEL(video, mfcc)
            prob_fake = torch.sigmoid(logit).item()

        label = "fake" if prob_fake >= 0.5 else "real"
        return label, prob_fake

    finally:
        # Cleanup temp file
        try:
            os.remove(video_path)
        except FileNotFoundError:
            pass

        # Explicitly release tensors to reduce peak RAM
        del video
        del mfcc
        gc.collect()
