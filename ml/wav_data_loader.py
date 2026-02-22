"""
WAV Dataset loader (loads raw .wav audio using split CSV).

Expected:
- split CSV has columns: video_id,label (and optionally video_path,av_category)
- WAV files stored under: <wav_root>/...

This loader is intentionally robust to naming differences, mirroring MFCCDataset:
it tries multiple candidate filenames based on:
- video_id
- video_path stem/name
- hashed ids (md5/sha1/sha256, full and 16-char truncations)

Returns:
- x: torch.FloatTensor [1, T] (mono waveform)
- y: torch.FloatTensor scalar (0/1)
- video_id: str

Dependencies:
    pip install numpy pandas torch
    pip install soundfile
(Alternative: torchaudio, but soundfile is lighter and Windows-friendly.)

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import soundfile as sf
except Exception as exc:
    raise ImportError(
        "Missing dependency: soundfile. Install with: python -m pip install soundfile"
    ) from exc


# Fixed length for batching (seconds * sample_rate)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SECONDS = 3
FIXED_SAMPLES = DEFAULT_SAMPLE_RATE * DEFAULT_SECONDS  # e.g., 48000


def _norm_path(s: str) -> str:
    return str(s).replace("\\", "/").strip()


def _to_mono(wav: np.ndarray) -> np.ndarray:
    # soundfile can return shape [T] or [T, C]
    if wav.ndim == 1:
        return wav
    if wav.ndim == 2:
        return wav.mean(axis=1)
    raise ValueError(f"Unexpected wav shape: {wav.shape}")


def _pad_or_crop_center(wav: np.ndarray, target_len: int) -> np.ndarray:
    T = wav.shape[0]
    if T == target_len:
        return wav
    if T < target_len:
        pad = target_len - T
        return np.pad(wav, (0, pad), mode="constant")
    # crop center
    start = (T - target_len) // 2
    return wav[start:start + target_len]


def _build_candidates(video_id: str, video_path_str: str) -> list[str]:
    """
    Builds candidate base names (without folder) similar to MFCCDataset.
    We'll try with .wav extension during lookup.
    """
    video_path_str = str(video_path_str or "")
    video_name = Path(video_path_str).name if video_path_str else ""
    video_stem = Path(video_path_str).stem if video_path_str else ""

    candidates = [video_id]

    # 1) CSV video_id as-is

    # 2) If video_id includes extension like ".mp4", try without it
    if video_id.lower().endswith(".mp4"):
        candidates.append(Path(video_id).stem)

    # 3) Try using original video filename/stem
    if video_name:
        candidates.append(video_name)   # e.g. 123.mp4
    if video_stem:
        candidates.append(video_stem)   # e.g. 123

    # 4) Hash candidates (md5 full)
    if video_path_str:
        candidates.append(hashlib.md5(video_path_str.encode("utf-8")).hexdigest())
        candidates.append(hashlib.md5(_norm_path(video_path_str).encode("utf-8")).hexdigest())
    if video_name:
        candidates.append(hashlib.md5(video_name.encode("utf-8")).hexdigest())
    candidates.append(hashlib.md5(video_id.encode("utf-8")).hexdigest())

    # 5) Truncated md5 (16)
    if video_path_str:
        candidates.append(hashlib.md5(video_path_str.encode("utf-8")).hexdigest()[:16])
        candidates.append(hashlib.md5(_norm_path(video_path_str).encode("utf-8")).hexdigest()[:16])
    if video_name:
        candidates.append(hashlib.md5(video_name.encode("utf-8")).hexdigest()[:16])
    candidates.append(hashlib.md5(video_id.encode("utf-8")).hexdigest()[:16])

    # 6) Truncated sha1/sha256 (16)
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

    # de-duplicate while preserving order
    out = []
    seen = set()
    for c in candidates:
        c = str(c).strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


class WavDataset(Dataset):
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

        if "video_id" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain at least: video_id,label")

        if not self.wav_root.exists():
            raise FileNotFoundError(f"WAV root folder not found: {self.wav_root}")

    def __len__(self):
        return len(self.df)

    def _find_wav_path(self, video_id: str, video_path_str: str) -> Path:
        candidates = _build_candidates(video_id, video_path_str)

        # We'll try these extensions; some pipelines may store .flac or .mp3, but start with wav
        exts = [".wav", ".WAV"]

        # 1) flat root: <wav_root>/<candidate>.wav
        for base in candidates:
            for ext in exts:
                p = self.wav_root / f"{base}{ext}"
                if p.exists():
                    return p

        # 2) recursive search (slower, but robust)
        # only try a small set to avoid heavy IO
        for base in candidates[:6]:
            for ext in exts:
                hits = list(self.wav_root.rglob(f"{base}{ext}"))
                if hits:
                    return hits[0]

        raise FileNotFoundError(
            "Missing WAV file. Tried (first few):\n  - " +
            "\n  - ".join(str(self.wav_root / f"{c}.wav") for c in candidates[:8])
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        label = int(row["label"])
        video_path_str = str(row.get("video_path", ""))

        wav_path = self._find_wav_path(video_id, video_path_str)

        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        wav = _to_mono(wav)

        # If sample rate differs, you can either resample or enforce extraction at 16kHz.
        # For coursework stability, we enforce: data should already be extracted at sample_rate.
        if int(sr) != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch for {wav_path}: got {sr}, expected {self.sample_rate}. "
                "Fix by re-extracting audio at 16kHz or add resampling."
            )

        wav = _pad_or_crop_center(wav, self.fixed_samples)

        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # [1, T]

        # per-sample normalization (like MFCCDataset)
        mean = x.mean()
        std = x.std().clamp(min=1e-6)
        x = (x - mean) / std

        y = torch.tensor(label, dtype=torch.float32)
        return x, y, video_id
