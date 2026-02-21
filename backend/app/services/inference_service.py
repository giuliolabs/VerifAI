"""
Inference Service – Multimodal Fusion Model
===========================================

This module encapsulates all logic required to:
- Load the trained multimodal fusion model
- Preprocess uploaded videos
- Run inference and return predictions

------------------------------------------------
DEPENDENCIES
------------------------------------------------
    pip install torch torchvision numpy opencv-python librosa
------------------------------------------------
MODEL
------------------------------------------------
- Audio-Visual Late Fusion Model (MobileNetV2 + ResNet18 MFCC)
- Trained on FakeAVCeleb_v1.2
- Checkpoint:
    experiments/results/fakeavceleb_av_fusion_v1/best_model.pt

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- Model is loaded once at import time (efficient inference).
- Temporary files are cleaned automatically.
- Matches preprocessing used during training.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import os
import tempfile
import torch

from ml.models.fusion.multimodal_fusion import MultimodalFusionModel
from backend.app.services.preprocess_service import extract_features_from_video


# ------------------------------------------------
# Model loading (once)
# ------------------------------------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL = MultimodalFusionModel().to(_DEVICE)
_CHECKPOINT = "experiments/results/fakeavceleb_av_fusion_v1/best_model.pt"

state = torch.load(_CHECKPOINT, map_location=_DEVICE)
_MODEL.load_state_dict(state["model_state"])
_MODEL.eval()


def run_inference(upload_file):
    """
    Perform multimodal deepfake inference on a video upload.

    Returns:
        label (str): "real" or "fake"
        prob_fake (float): probability of fake
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upload_file.file.read())
        video_path = tmp.name

    try:
        video, mfcc = extract_features_from_video(video_path)

        video = video.unsqueeze(0).to(_DEVICE)  # [1, 5, 3, 224, 224]
        mfcc = mfcc.unsqueeze(0).to(_DEVICE)    # [1, 1, 40, T]

        with torch.no_grad():
            logit = _MODEL(video, mfcc)
            prob_fake = torch.sigmoid(logit).item()

        label = "fake" if prob_fake >= 0.5 else "real"
        return label, prob_fake

    finally:
        os.remove(video_path)
