"""
WAV Dataset Loader (Raw Audio) – VerifAI
=========================================

This module implements a robust PyTorch Dataset for loading raw
.wav audio files based on a split CSV file.

Expected:
- Split CSV must contain at least: video_id, label
- WAV files stored under: <wav_root>/...

Design goals:
- Robust filename resolution (mirrors MFCCDataset logic)
- Fixed-length audio for batching stability
- Strict sample rate enforcement for reproducibility
- Per-sample normalization

This loader supports ablation experiments between:
- MFCC-based models
- Raw waveform models

Dependencies:
    pip install numpy pandas torch
    pip install soundfile

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# soundfile is lightweight and stable across platforms
try:
    import soundfile as sf
except Exception as exc:
    raise ImportError(
        "Missing dependency: soundfile. Install with: python -m pip install soundfile"
    ) from exc


# ------------------------------------------------
# Global Configuration
# ------------------------------------------------

# Default extraction settings
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SECONDS = 3

# Fixed number of samples per example
FIXED_SAMPLES = DEFAULT_SAMPLE_RATE * DEFAULT_SECONDS


# ------------------------------------------------
# Utility Functions
# ------------------------------------------------

def _norm_path(s: str) -> str:
    """Normalize path formatting for hashing consistency."""
    return str(s).replace("\\", "/").strip()


def _to_mono(wav: np.ndarray) -> np.ndarray:
    """
    Convert multi-channel audio to mono by averaging channels.
    """
    if wav.ndim == 1:
        return wav
    if wav.ndim == 2:
        return wav.mean(axis=1)
    raise ValueError(f"Unexpected wav shape: {wav.shape}")


def _pad_or_crop_center(wav: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ensure waveform has fixed length.
    - Pad with zeros if too short
    - Crop centrally if too long
    """
    T = wav.shape[0]

    if T == target_len:
        return wav

    if T < target_len:
        pad = target_len - T
        return np.pad(wav, (0, pad), mode="constant")

    # Center crop for longer waveforms
    start = (T - target_len) // 2
    return wav[start:start + target_len]


def _build_candidates(video_id: str, video_path_str: str) -> list[str]:
    """
    Generate possible filename candidates for locating WAV files.

    This mirrors MFCCDataset logic and improves robustness across
    different extraction pipelines.
    """

    video_path_str = str(video_path_str or "")
    video_name = Path(video_path_str).name if video_path_str else ""
    video_stem = Path(video_path_str).stem if video_path_str else ""

    candidates = [video_id]

    # If video_id contains extension, try stem version
    if video_id.lower().endswith(".mp4"):
        candidates.append(Path(video_id).stem)

    # Include original filename variations
    if video_name:
        candidates.append(video_name)
    if video_stem:
        candidates.append(video_stem)

    # Hash-based fallback strategies
    if video_path_str:
        candidates.append(hashlib.md5(video_path_str.encode("utf-8")).hexdigest())
        candidates.append(hashlib.md5(_norm_path(video_path_str).encode("utf-8")).hexdigest())

    if video_name:
        candidates.append(hashlib.md5(video_name.encode("utf-8")).hexdigest())

    candidates.append(hashlib.md5(video_id.encode("utf-8")).hexdigest())

    # Truncated hashes (16 characters)
    if video_path_str:
        candidates.append(hashlib.md5(video_path_str.encode("utf-8")).hexdigest()[:16])
        candidates.append(hashlib.md5(_norm_path(video_path_str).encode("utf-8")).hexdigest()[:16])

    if video_name:
        candidates.append(hashlib.md5(video_name.encode("utf-8")).hexdigest()[:16])

    candidates.append(hashlib.md5(video_id.encode("utf-8")).hexdigest()[:16])

    # Additional sha1/sha256 truncated hashes
    if video_path_str:
        norm_path = _norm_path(video_path_str)
        candidates.append(hashlib.sha1(video_path_str.encode("utf-8")).hexdigest()[:16])
        candidates.append(hashlib.sha1(norm_path.encode("utf-8")).hexdigest()[:16])
        candidates.append(hashlib.sha256(video_path_str.encode("utf-8")).hexdigest()[:16])
        candidates.append(hashlib.sha256(norm_path.encode("utf-8")).hexdigest()[:16])

    if video_name:
        candidates.append(hashlib.sha1(video_name.encode("utf-8")).hexdigest()[:16])
        candidates.append(hashlib.sha256(video_name.encode("utf-8")).hexdigest()[:16])

    candidates.append(hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:16])
    candidates.append(hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:16])

    # Remove duplicates while preserving order
    out = []
    seen = set()
    for c in candidates:
        c = str(c).strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    return out


# ------------------------------------------------
# Dataset Class
# ------------------------------------------------

class WavDataset(Dataset):
    """
    PyTorch Dataset for loading raw waveform audio files.

    Returns:
        x: [1, T] normalized waveform tensor
        y: scalar float label (0 or 1)
        video_id: identifier string
    """

    def __init__(
            self,
            split_csv: str,
            wav_root: str,
            sample_rate: int = DEFAULT_SAMPLE_RATE,
            seconds: int = DEFAULT_SECONDS,
            strict_sr: bool = True,
    ):
        self.df = pd.read_csv(split_csv)
        self.wav_root = Path(wav_root)
        self.sample_rate = int(sample_rate)
        self.fixed_samples = int(sample_rate) * int(seconds)
        self.strict_sr = bool(strict_sr)

        # Basic CSV validation
        if "video_id" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain at least: video_id,label")

        if not self.wav_root.exists():
            raise FileNotFoundError(f"WAV root folder not found: {self.wav_root}")

    def __len__(self):
        return len(self.df)

    def _find_wav_path(self, video_id: str, video_path_str: str) -> Path:
        """
        Locate WAV file using candidate matching strategy.
        """

        candidates = _build_candidates(video_id, video_path_str)
        exts = [".wav", ".WAV"]

        # First attempt: flat directory lookup
        for base in candidates:
            for ext in exts:
                p = self.wav_root / f"{base}{ext}"
                if p.exists():
                    return p

        # Fallback: recursive search
        for base in candidates[:6]:
            for ext in exts:
                hits = list(self.wav_root.rglob(f"{base}{ext}"))
                if hits:
                    return hits[0]

        raise FileNotFoundError(
            "Missing WAV file. Tried:\n  - " +
            "\n  - ".join(str(self.wav_root / f"{c}.wav") for c in candidates[:8])
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_id = str(row["video_id"])
        label = int(row["label"])
        video_path_str = str(row.get("video_path", ""))

        wav_path = self._find_wav_path(video_id, video_path_str)

        # Load waveform
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        wav = _to_mono(wav)

        # Enforce consistent sample rate for stability
        if int(sr) != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch for {wav_path}: got {sr}, expected {self.sample_rate}."
            )

        # Fix waveform length for batching
        wav = _pad_or_crop_center(wav, self.fixed_samples)

        # Convert to tensor and add channel dimension
        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # [1, T]

        # Per-sample normalization (zero mean, unit variance)
        mean = x.mean()
        std = x.std().clamp(min=1e-6)
        x = (x - mean) / std

        y = torch.tensor(label, dtype=torch.float32)

        return x, y, video_id