"""
Inference Service – Multimodal Deepfake Detection (AV + Visual Fallback)
=========================================================================

This module encapsulates all logic required to:
- Load trained deepfake detection models
- Preprocess uploaded videos
- Detect whether usable audio is available
- Dynamically choose the most appropriate model at runtime
- Run inference and return a fake probability

------------------------------------------------
DEPENDENCIES
------------------------------------------------
    pip install torch torchvision numpy opencv-python librosa timm

------------------------------------------------
DEPLOYED INFERENCE STRATEGY
------------------------------------------------
Primary model:
- fakeavceleb_av_fusion_v1
- Type: Audio-Visual (AV) fusion
- Dataset: FakeAVCeleb_v1.2

Fallback model:
- ffpp_c23_vit_baseline
- Type: Visual-only
- Dataset: FaceForensics++ C23
- Used only when the uploaded video has no usable audio track

------------------------------------------------
RUNTIME DECISION LOGIC
------------------------------------------------
If the uploaded video contains usable audio:
    -> run the AV fusion model

If the uploaded video has no audio, or audio features are empty:
    -> run the visual-only fallback model

------------------------------------------------
OUTPUT INTERPRETATION
------------------------------------------------
Both models are expected to output a single logit.

The logit is converted to a fake probability using:
    prob_fake = sigmoid(logit)

Final class decision:
    label = "fake" if prob_fake >= threshold else "real"

For the visual-only fallback:
- multiple extracted frames are scored independently
- frame logits are averaged
- sigmoid is applied once to the averaged logit
This is more stable than relying on a single frame.

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- Models are loaded once at import time for efficient inference.
- Temporary uploaded files are cleaned automatically.
- The AV model is prioritized because it uses both visual and audio evidence.
- Silent or audio-stripped videos are handled explicitly via a visual fallback.
- The visual fallback model is frame-based, so multiple representative frames
  are evaluated and their logits are averaged before making a final decision.
- This avoids unreliable behavior caused by feeding empty audio features
  into a multimodal fusion model and improves stability for silent clips.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

import os
import tempfile

import torch
from fastapi import HTTPException

from ml.models.fusion.multimodal_fusion import MultimodalFusionModel
from ml.models.video.vit import build_vit_binary
from backend.app.services.preprocess_service import extract_features_from_video


# ------------------------------------------------
# Configuration
# ------------------------------------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_THRESHOLD = 0.5

_AV_CHECKPOINT = "experiments/results/fakeavceleb_av_fusion_v1/best_model.pt"
_VISUAL_CHECKPOINT = "experiments/results/ffpp_c23_vit_baseline/best_model.pt"


# ------------------------------------------------
# Load primary Audio-Visual fusion model
# ------------------------------------------------

av_model = MultimodalFusionModel().to(_DEVICE)
av_state = torch.load(_AV_CHECKPOINT, map_location=_DEVICE, weights_only=False)
av_model.load_state_dict(av_state["model_state"])
av_model.eval()


# ------------------------------------------------
# Load visual-only fallback model
# ------------------------------------------------

visual_model = build_vit_binary(
    model_name="vit_base_patch16_224",
    pretrained=False,
    img_size=224,
).to(_DEVICE)

visual_state = torch.load(_VISUAL_CHECKPOINT, map_location=_DEVICE, weights_only=False)
visual_model.load_state_dict(visual_state["model_state"])
visual_model.eval()


# ------------------------------------------------
# Inference
# ------------------------------------------------

def run_inference(upload_file):
    """
    Perform deepfake inference on an uploaded video.

    Returns:
        label (str): "real" or "fake"
        prob_fake (float): probability that the input is fake
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upload_file.file.read())
        video_path = tmp.name

    try:
        video, mfcc = extract_features_from_video(video_path)

        video = video.unsqueeze(0).to(_DEVICE)   # [1, T, 3, 224, 224]
        mfcc = mfcc.unsqueeze(0).to(_DEVICE)     # [1, 1, 40, 200]

        if video.shape[1] == 0:
            raise HTTPException(status_code=400, detail="No valid video frames could be extracted.")

        has_audio = not torch.all(mfcc == 0)

        with torch.no_grad():
            if has_audio:
                logit = av_model(video, mfcc)
                logit_value = float(logit.squeeze().item())
                prob_fake = float(torch.sigmoid(logit.squeeze()).item())

            else:
                all_frame_logits = []

                for frame_index in range(video.shape[1]):
                    frame = video[:, frame_index, :, :, :]   # [1, 3, 224, 224]
                    frame_logit = visual_model(frame)
                    all_frame_logits.append(float(frame_logit.squeeze().item()))

                logit_value = sum(all_frame_logits) / len(all_frame_logits)
                prob_fake = float(torch.sigmoid(torch.tensor(logit_value, device=_DEVICE)).item())

        label = "fake" if prob_fake >= _THRESHOLD else "real"
        return label, prob_fake

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
