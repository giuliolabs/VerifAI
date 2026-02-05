"""
Preprocessing Service – API Inference
====================================

This module provides lightweight preprocessing utilities used
by the FastAPI inference pipeline.

It mirrors the preprocessing used during training:
- Extracts 5 frames from the video
- Extracts audio and computes MFCC features

------------------------------------------------
DEPENDENCIES
------------------------------------------------
    pip install torch opencv-python librosa moviepy numpy

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- This preprocessing is intentionally minimal and deterministic.
- Designed for inference only (not training).
- Matches FakeAVCeleb multimodal fusion setup.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import cv2
import librosa
from moviepy import VideoFileClip
import numpy as np
import torch
import tempfile
import os


def extract_features_from_video(video_path: str, n_frames: int = 5):
    """
    Extract video frames and MFCC features from a video file.

    Returns:
        video_tensor: torch.FloatTensor [5, 3, 224, 224]
        mfcc_tensor:  torch.FloatTensor [1, 40, T]
    """

    # -------------------------
    # Video → Frames
    # -------------------------
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Invalid or empty video file")

    indices = np.linspace(0, total_frames - 1, n_frames).astype(int)
    idx_set = set(indices)

    current = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current in idx_set:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))  # CHW
            frames.append(frame)

        current += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video")

    video_tensor = torch.from_numpy(np.stack(frames)).float()

    # -------------------------
    # Audio → MFCC
    # -------------------------
    mfcc_tensor = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        audio_path = tmp_audio.name

    try:
        clip = VideoFileClip(video_path)

        if clip.audio is None:
            # No audio track → silent placeholder
            mfcc_tensor = torch.zeros((1, 40, 1), dtype=torch.float32)
        else:
            clip.audio.write_audiofile(
                audio_path,
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                logger=None,
            )

            y, sr = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_tensor = torch.from_numpy(mfcc).float().unsqueeze(0)

        clip.close()

    finally:
        # Clean up temp audio file safely (Windows-safe)
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except (PermissionError, FileNotFoundError):
            pass

    return video_tensor, mfcc_tensor
