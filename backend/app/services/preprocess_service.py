"""
Preprocessing Service – API Inference
====================================

This module provides lightweight preprocessing utilities used
by the FastAPI inference pipeline.

It mirrors the preprocessing used during training:
- Extracts exactly 5 frames from the video
- Resizes frames to 224x224
- Applies ImageNet normalization to frames
- Extracts audio and computes 40 MFCC features
- Pads/trims MFCC time dimension to a fixed length
- Applies per-sample MFCC normalization

------------------------------------------------
DEPENDENCIES
------------------------------------------------
    pip install torch opencv-python librosa moviepy numpy

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- This preprocessing is intentionally deterministic.
- Designed for inference only (not training).
- Matches the FakeAVCeleb multimodal fusion setup as closely as possible.
- Silent videos are supported by returning a zero MFCC placeholder.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

import os
import tempfile

import cv2
import librosa
import numpy as np
import torch
from moviepy import VideoFileClip


# ------------------------------------------------
# Configuration
# ------------------------------------------------

IMAGE_SIZE = 224
N_FRAMES = 5
N_MFCC = 40
MFCC_MAX_LEN = 200

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _normalize_frame(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Convert frame to CHW float32 tensor-like numpy array and apply
    the same normalization used during training.
    """
    frame = frame_rgb.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return frame


def _pad_or_trim_mfcc(mfcc: np.ndarray, max_len: int) -> np.ndarray:
    """
    Ensure MFCC has shape [N_MFCC, max_len].
    """
    time_steps = mfcc.shape[1]

    if time_steps > max_len:
        mfcc = mfcc[:, :max_len]
    elif time_steps < max_len:
        pad_cols = max_len - time_steps
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_cols)), mode="constant")

    return mfcc


def _normalize_mfcc(mfcc_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply per-sample normalization exactly like training:
        x = (x - mean) / std
    """
    mean = mfcc_tensor.mean()
    std = mfcc_tensor.std().clamp(min=1e-6)
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
    Extract video frames and MFCC features from a video file.

    Returns:
        video_tensor: torch.FloatTensor [n_frames, 3, image_size, image_size]
        mfcc_tensor:  torch.FloatTensor [1, n_mfcc, mfcc_max_len]
    """

    # ------------------------------------------------
    # Video -> Frames
    # ------------------------------------------------

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid or empty video file")

    target_indices = np.linspace(0, total_frames - 1, n_frames).astype(int).tolist()
    frames: list[np.ndarray] = []

    for frame_index in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.resize(frame, (image_size, image_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _normalize_frame(frame)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video")

    # If a few frames fail to load, repeat the last valid frame until we reach n_frames
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    video_tensor = torch.from_numpy(np.stack(frames[:n_frames])).float()

    # ------------------------------------------------
    # Audio -> MFCC
    # ------------------------------------------------

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        audio_path = tmp_audio.name

    clip = None
    try:
        clip = VideoFileClip(video_path)

        if clip.audio is None:
            mfcc_tensor = torch.zeros((1, n_mfcc, mfcc_max_len), dtype=torch.float32)
        else:
            clip.audio.write_audiofile(
                audio_path,
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                logger=None,
            )

            y, sr = librosa.load(audio_path, sr=16000)

            if y is None or len(y) == 0:
                mfcc_tensor = torch.zeros((1, n_mfcc, mfcc_max_len), dtype=torch.float32)
            else:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                mfcc = _pad_or_trim_mfcc(mfcc, mfcc_max_len)

                mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
                mfcc_tensor = _normalize_mfcc(mfcc_tensor)

    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass

        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except (PermissionError, FileNotFoundError):
            pass

    return video_tensor, mfcc_tensor