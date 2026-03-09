"""
Preprocessing Service – API Inference
=====================================

This module implements the preprocessing pipeline used during inference
in the VerifAI backend. Its purpose is to transform raw video files into
structured tensors that match the format expected by the trained models.

The implementation mirrors the preprocessing strategy used during training
to ensure consistency between training and deployment environments.

It performs:
- Deterministic extraction of exactly 5 frames
- Resizing frames to 224x224 resolution
- ImageNet normalization for visual backbone compatibility
- Audio extraction and computation of 40 MFCC coefficients
- Fixed-length padding of MFCC features
- Per-sample MFCC normalization

Silent or invalid audio is safely handled using a zero placeholder tensor,
allowing the hybrid system to fall back to visual-only inference.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
    pip install torch opencv-python librosa moviepy numpy
------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- Preprocessing is deterministic for reproducibility.
- Designed strictly for inference.
- Matches the FakeAVCeleb multimodal fusion setup.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Used for temporary audio file management and cleanup
import os
import tempfile

# OpenCV for frame extraction from video
import cv2

# Librosa for audio loading and MFCC computation
import librosa

# NumPy for array manipulation and numerical processing
import numpy as np

# PyTorch for tensor creation and normalization
import torch

# MoviePy for extracting audio from video files
from moviepy import VideoFileClip


# ------------------------------------------------
# Configuration
# ------------------------------------------------

# Target frame resolution expected by the visual model
IMAGE_SIZE = 224

# Number of frames extracted per video (deterministic sampling)
N_FRAMES = 5

# Number of MFCC coefficients used for audio representation
N_MFCC = 40

# Fixed temporal length for MFCC features
MFCC_MAX_LEN = 200

# ImageNet normalization constants (required for Xception backbone)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _normalize_frame(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Convert frame to CHW float32 format and apply ImageNet normalization.
    This ensures compatibility with the pretrained visual backbone.
    """

    # Scale pixel values from [0,255] to [0,1]
    frame = frame_rgb.astype(np.float32) / 255.0

    # Convert layout from HWC (OpenCV default) to CHW (PyTorch format)
    frame = np.transpose(frame, (2, 0, 1))

    # Apply standard ImageNet normalization
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return frame


def _pad_or_trim_mfcc(mfcc: np.ndarray, max_len: int) -> np.ndarray:
    """
    Ensure MFCC has shape [N_MFCC, max_len].
    This keeps audio inputs consistent across different video durations.
    """

    time_steps = mfcc.shape[1]

    # Trim longer sequences
    if time_steps > max_len:
        mfcc = mfcc[:, :max_len]

    # Pad shorter sequences with zeros
    elif time_steps < max_len:
        pad_cols = max_len - time_steps
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_cols)), mode="constant")

    return mfcc


def _normalize_mfcc(mfcc_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply per-sample standardization:
        x = (x - mean) / std

    This matches the normalization used during training.
    """

    mean = mfcc_tensor.mean()
    std = mfcc_tensor.std().clamp(min=1e-6)  # Avoid division by zero
    mfcc_tensor = (mfcc_tensor - mean) / std
    return mfcc_tensor


def extract_features_from_video(
    video_path: str,
    n_frames: int = N_FRAMES,
    image_size: int = IMAGE_SIZE,
    n_mfcc: int = N_MFCC,
    mfcc_max_len: int = MFCC_MAX_LEN,
):
    """
    Extract visual and audio features from a video file.

    Returns:
        video_tensor: torch.FloatTensor [n_frames, 3, image_size, image_size]
        mfcc_tensor:  torch.FloatTensor [1, n_mfcc, mfcc_max_len]
    """

    # ------------------------------------------------
    # Video -> Frame Extraction
    # ------------------------------------------------

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Basic validation to ensure the file contains readable frames
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid or empty video file")

    # Evenly sample frame indices across the video duration
    target_indices = np.linspace(0, total_frames - 1, n_frames).astype(int).tolist()

    frames: list[np.ndarray] = []

    for frame_index in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            continue

        # Resize to model input resolution
        frame = cv2.resize(frame, (image_size, image_size))

        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply normalization
        frame = _normalize_frame(frame)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video")

    # If fewer frames are extracted than expected,
    # duplicate the last valid frame to maintain fixed input size
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    # Stack frames into a single tensor
    video_tensor = torch.from_numpy(np.stack(frames[:n_frames])).float()

    # ------------------------------------------------
    # Audio -> MFCC Extraction
    # ------------------------------------------------

    # Create temporary WAV file for audio extraction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        audio_path = tmp_audio.name

    clip = None
    try:
        clip = VideoFileClip(video_path)

        # If the video contains no audio stream, return a zero tensor
        if clip.audio is None:
            mfcc_tensor = torch.zeros((1, n_mfcc, mfcc_max_len), dtype=torch.float32)

        else:
            # Export audio to 16kHz WAV format (standard for MFCC extraction)
            clip.audio.write_audiofile(
                audio_path,
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                logger=None,
            )

            y, sr = librosa.load(audio_path, sr=16000)

            # Handle corrupted or empty audio safely
            if y is None or len(y) == 0:
                mfcc_tensor = torch.zeros((1, n_mfcc, mfcc_max_len), dtype=torch.float32)

            else:
                # Compute MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

                # Enforce fixed temporal dimension
                mfcc = _pad_or_trim_mfcc(mfcc, mfcc_max_len)

                mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

                # Apply per-sample normalization
                mfcc_tensor = _normalize_mfcc(mfcc_tensor)

    finally:
        # Ensure video clip is properly closed
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass

        # Clean up temporary audio file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except (PermissionError, FileNotFoundError):
            pass

    return video_tensor, mfcc_tensor