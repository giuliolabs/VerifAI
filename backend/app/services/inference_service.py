"""
Inference Service – Final VerifAI Runtime
=========================================

Runtime strategy:
- If usable audio is available:
    -> use final_hybrid_av
- Otherwise:
    -> use final_visual_all_datasets_xception

This service:
- loads the final trained models once
- preprocesses uploaded videos
- chooses hybrid or visual-only fallback automatically
- returns label + fake probability + inference mode
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import joblib
import torch
import torch.nn.functional as f
from fastapi import HTTPException

from ml.models.video.xception import build_xception_binary
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary
from backend.app.services.preprocess_service import extract_features_from_video


# ------------------------------------------------
# Configuration
# ------------------------------------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FINAL_HYBRID_DIR = Path("experiments/results/final_hybrid_av")
FINAL_VISUAL_DIR = Path("experiments/results/final_visual_all_datasets_xception")
FINAL_AUDIO_DIR = Path("experiments/results/fakeavceleb_audio_resnet_baseline")

_VISUAL_CHECKPOINT = FINAL_VISUAL_DIR / "best_model.pt"
_AUDIO_CHECKPOINT = FINAL_AUDIO_DIR / "best_model.pt"
_FUSION_MODEL_PATH = FINAL_HYBRID_DIR / "fusion_model.joblib"
_FUSION_META_PATH = FINAL_HYBRID_DIR / "fusion_meta.json"

_VISUAL_ONLY_THRESHOLD = 0.5


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def _torch_load_compat(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _validate_artifacts() -> None:
    missing = []
    for path in [
        _VISUAL_CHECKPOINT,
        _AUDIO_CHECKPOINT,
        _FUSION_MODEL_PATH,
        _FUSION_META_PATH,
    ]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        missing_text = "\n".join(missing)
        raise RuntimeError(f"Missing required inference artifacts:\n{missing_text}")


_validate_artifacts()


# ------------------------------------------------
# Load final visual backbone
# ------------------------------------------------

visual_model = build_xception_binary(pretrained=False).to(_DEVICE)
visual_state = _torch_load_compat(_VISUAL_CHECKPOINT, _DEVICE)
visual_model.load_state_dict(visual_state["model_state"])
visual_model.eval()


# ------------------------------------------------
# Load final audio backbone
# ------------------------------------------------

audio_model = build_mfcc_resnet18_binary(pretrained=False).to(_DEVICE)
audio_state = _torch_load_compat(_AUDIO_CHECKPOINT, _DEVICE)
audio_model.load_state_dict(audio_state["model_state"])
audio_model.eval()


# ------------------------------------------------
# Load final fusion model + threshold
# ------------------------------------------------

fusion_model = joblib.load(_FUSION_MODEL_PATH)
fusion_meta = json.loads(_FUSION_META_PATH.read_text(encoding="utf-8"))
HYBRID_THRESHOLD = float(fusion_meta.get("best_threshold", 0.5))


# ------------------------------------------------
# Internal scoring
# ------------------------------------------------

@torch.no_grad()
def _score_visual_only(video: torch.Tensor) -> float:
    """
    video: [1, T, 3, 224, 224]
    Returns fake probability from visual model only.
    """
    if video.ndim != 5 or video.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No valid video frames could be extracted.")

    batch_size, time_steps, channels, height, width = video.shape
    frames = video.view(batch_size * time_steps, channels, height, width)
    frames = f.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)

    logits = visual_model(frames).view(batch_size, time_steps)
    mean_logit = logits.mean(dim=1)
    prob_fake = float(torch.sigmoid(mean_logit.squeeze()).item())
    return prob_fake


@torch.no_grad()
def _score_hybrid(video: torch.Tensor, mfcc: torch.Tensor) -> float:
    """
    video: [1, T, 3, 224, 224]
    mfcc:  [1, 1, 40, 200]
    Returns fake probability from late-fusion hybrid model.
    """
    batch_size, time_steps, channels, height, width = video.shape

    frames = video.view(batch_size * time_steps, channels, height, width)
    frames = f.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)

    visual_logits = visual_model(frames).view(batch_size, time_steps)
    visual_prob = torch.sigmoid(visual_logits).mean(dim=1)

    audio_logits = audio_model(mfcc).squeeze(1)
    audio_prob = torch.sigmoid(audio_logits)

    fusion_features = torch.stack([visual_prob, audio_prob], dim=1).cpu().numpy()
    prob_fake = float(fusion_model.predict_proba(fusion_features)[:, 1][0])
    return prob_fake


# ------------------------------------------------
# Public inference
# ------------------------------------------------

def run_inference(upload_file):
    """
    Perform deepfake inference on an uploaded video.

    Returns:
        dict with:
        - label
        - prob_fake
        - mode
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

        has_audio = mfcc.numel() > 0 and not torch.all(mfcc == 0)

        if has_audio:
            prob_fake = _score_hybrid(video, mfcc)
            threshold = HYBRID_THRESHOLD
            mode = "hybrid_av"
        else:
            prob_fake = _score_visual_only(video)
            threshold = _VISUAL_ONLY_THRESHOLD
            mode = "visual_only_fallback"

        label = "fake" if prob_fake >= threshold else "real"

        # Convert to percentage with 2 decimal places
        prob_percent = round(prob_fake * 100, 2)
        prob_string = f"{prob_percent:.2f}%"

        return {
            "label": label,
            "prob_fake": prob_string,
            "mode": mode,
        }

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
