"""
Inference Service – Final VerifAI Runtime
=========================================

This module contains the main runtime inference logic for the VerifAI
deepfake detection system. Its role is to load the final trained models,
prepare uploaded video data for prediction, and automatically choose the
most suitable inference strategy depending on whether valid audio features
are available. This ensures the system remains usable even when audio is
missing or unusable.

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

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Import JSON to read metadata such as the saved hybrid threshold
import json

# Import os to handle file existence checks and deletion of temporary files
import os

# Import tempfile to safely create temporary files for uploaded videos
import tempfile

# Import Path for cleaner and more reliable filesystem path handling
from pathlib import Path

# Import joblib to load the trained fusion model
import joblib

# Import torch for tensor operations and model inference
import torch

# Import torch.nn.functional for resizing frame tensors
import torch.nn.functional as f

# Import HTTPException to return meaningful API errors when inference fails
from fastapi import HTTPException

# Import the final visual deepfake classifier architecture
from ml.models.video.xception import build_xception_binary

# Import the final audio classifier architecture based on MFCC features
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary

# Import the preprocessing pipeline used to extract video frames and audio features
from backend.app.services.preprocess_service import extract_features_from_video


# ------------------------------------------------
# Configuration
# ------------------------------------------------

# Select GPU if available for faster inference, otherwise fall back to CPU
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the directories containing the final trained model artifacts
FINAL_HYBRID_DIR = Path("experiments/results/final_hybrid_av")
FINAL_VISUAL_DIR = Path("experiments/results/final_visual_all_datasets_xception")
FINAL_AUDIO_DIR = Path("experiments/results/fakeavceleb_audio_resnet_baseline")

# Define the exact paths to the saved visual, audio, and fusion model files
_VISUAL_CHECKPOINT = FINAL_VISUAL_DIR / "best_model.pt"
_AUDIO_CHECKPOINT = FINAL_AUDIO_DIR / "best_model.pt"
_FUSION_MODEL_PATH = FINAL_HYBRID_DIR / "fusion_model.joblib"
_FUSION_META_PATH = FINAL_HYBRID_DIR / "fusion_meta.json"

# Default threshold used when only the visual model is available
_VISUAL_ONLY_THRESHOLD = 0.5


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def _torch_load_compat(path: Path, device: str):
    """
    Load a PyTorch checkpoint in a way that is compatible across
    different torch versions.
    """

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _validate_artifacts() -> None:
    """
    Ensure that all required trained model files exist before
    the service starts accepting inference requests.
    """

    # Store any missing files so they can be reported together
    missing = []

    # Check whether each required artifact exists on disk
    for path in [
        _VISUAL_CHECKPOINT,
        _AUDIO_CHECKPOINT,
        _FUSION_MODEL_PATH,
        _FUSION_META_PATH,
    ]:
        if not path.exists():
            missing.append(str(path))

    # If any required files are missing, stop startup early with
    # a clear error message rather than failing later during inference
    if missing:
        missing_text = "\n".join(missing)
        raise RuntimeError(f"Missing required inference artifacts:\n{missing_text}")


# Validate model files as soon as the module is loaded
# This helps catch deployment issues immediately
_validate_artifacts()


# ------------------------------------------------
# Load final visual backbone
# ------------------------------------------------

# Build the visual model architecture and move it to the chosen device
visual_model = build_xception_binary(pretrained=False).to(_DEVICE)

# Load the saved trained weights for the visual model
visual_state = _torch_load_compat(_VISUAL_CHECKPOINT, _DEVICE)

# Restore the learned parameters into the model
visual_model.load_state_dict(visual_state["model_state"])

# Set the model to evaluation mode to disable training-specific behavior
visual_model.eval()


# ------------------------------------------------
# Load final audio backbone
# ------------------------------------------------

# Build the audio model architecture and move it to the chosen device
audio_model = build_mfcc_resnet18_binary(pretrained=False).to(_DEVICE)

# Load the saved trained weights for the audio model
audio_state = _torch_load_compat(_AUDIO_CHECKPOINT, _DEVICE)

# Restore the learned parameters into the model
audio_model.load_state_dict(audio_state["model_state"])

# Set the model to evaluation mode for inference consistency
audio_model.eval()


# ------------------------------------------------
# Load final fusion model + threshold
# ------------------------------------------------

# Load the late-fusion model that combines visual and audio predictions
fusion_model = joblib.load(_FUSION_MODEL_PATH)

# Load metadata for the fusion model, including the best decision threshold
fusion_meta = json.loads(_FUSION_META_PATH.read_text(encoding="utf-8"))

# Use the stored best threshold if available, otherwise default to 0.5
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

    # Validate that the tensor has the expected dimensions
    # and that at least one frame was successfully extracted
    if video.ndim != 5 or video.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No valid video frames could be extracted.")

    # Unpack the tensor shape for clarity
    batch_size, time_steps, channels, height, width = video.shape

    # Flatten the temporal dimension so each frame can be processed
    # individually by the image-based visual backbone
    frames = video.view(batch_size * time_steps, channels, height, width)

    # Resize frames to 299x299 because Xception expects this input size
    frames = f.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)

    # Run the visual model on all frames and reshape back into video form
    logits = visual_model(frames).view(batch_size, time_steps)

    # Average frame-level logits to obtain one score for the whole video
    mean_logit = logits.mean(dim=1)

    # Apply sigmoid to convert the final logit into a probability value
    prob_fake = float(torch.sigmoid(mean_logit.squeeze()).item())
    return prob_fake


@torch.no_grad()
def _score_hybrid(video: torch.Tensor, mfcc: torch.Tensor) -> float:
    """
    video: [1, T, 3, 224, 224]
    mfcc:  [1, 1, 40, 200]
    Returns fake probability from late-fusion hybrid model.
    """

    # Extract dimensions from the video tensor
    batch_size, time_steps, channels, height, width = video.shape

    # Flatten frames so they can be passed through the visual backbone
    frames = video.view(batch_size * time_steps, channels, height, width)

    # Resize frames to match the visual model input requirements
    frames = f.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)

    # Generate frame-level visual predictions and average them across time
    visual_logits = visual_model(frames).view(batch_size, time_steps)
    visual_prob = torch.sigmoid(visual_logits).mean(dim=1)

    # Run the audio model on the MFCC feature representation
    audio_logits = audio_model(mfcc).squeeze(1)
    audio_prob = torch.sigmoid(audio_logits)

    # Combine both modality probabilities into a 2-feature input for the late-fusion classifier
    fusion_features = torch.stack([visual_prob, audio_prob], dim=1).cpu().numpy()

    # Take the probability of the "fake" class from the fusion model
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

    # Save the uploaded file to a temporary location because
    # the preprocessing pipeline expects a file path as input
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(upload_file.file.read())
        video_path = tmp.name

    try:
        # Extract visual frames and MFCC audio features from the uploaded video
        video, mfcc = extract_features_from_video(video_path)

        # Add a batch dimension and move tensors to the selected device
        video = video.unsqueeze(0).to(_DEVICE)   # [1, T, 3, 224, 224]
        mfcc = mfcc.unsqueeze(0).to(_DEVICE)     # [1, 1, 40, 200]

        # Ensure at least one valid frame exists before scoring
        if video.shape[1] == 0:
            raise HTTPException(status_code=400, detail="No valid video frames could be extracted.")

        # Check whether meaningful audio is available.
        # A tensor of all zeros is treated as unusable or missing audio.
        has_audio = mfcc.numel() > 0 and not torch.all(mfcc == 0)

        # Use the hybrid pipeline when valid audio exists,
        # otherwise fall back to visual-only inference
        if has_audio:
            prob_fake = _score_hybrid(video, mfcc)
            threshold = HYBRID_THRESHOLD
            mode = "hybrid_av"
        else:
            prob_fake = _score_visual_only(video)
            threshold = _VISUAL_ONLY_THRESHOLD
            mode = "visual_only_fallback"

        # Convert probability into the final predicted class label
        label = "fake" if prob_fake >= threshold else "real"

        # Convert probability to a percentage string with 2 decimal places
        # for clearer API output
        prob_percent = round(prob_fake * 100, 2)
        prob_string = f"{prob_percent:.2f}%"

        # Return a clean structured response for the API layer
        return {
            "label": label,
            "prob_fake": prob_string,
            "mode": mode,
        }

    finally:
        # Always remove the temporary file after processing
        # to avoid unnecessary file accumulation on the server
        if os.path.exists(video_path):
            os.remove(video_path)