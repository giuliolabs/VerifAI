"""
Multimodal (Audio + Video) Dataset Loader for FakeAVCeleb_v1.2
==============================================================

This module defines the multimodal dataset loader used by VerifAI for
audio-visual deepfake experiments on FakeAVCeleb_v1.2. It is designed
to match the exact preprocessing pipeline already used in the project,
so frame folders and MFCC files can be loaded directly without extra
conversion logic.

This loader matches YOUR preprocessing scripts exactly:

- Frames saved by:
  pipelines/video/extract_frames_from_csv.py
    safe_id = sha1(video_id)[:16]
    <frames_root>/<split>/<safe_id>/frame_000.jpg ... frame_004.jpg

- Audio saved by:
  pipelines/audio/extract_audio.py
    safe_id = sha1(video_id)[:16]
    <audio_root>/<split>/<safe_id>.wav

- MFCC saved by:
  pipelines/audio/audio_features.py
    out_path = <mfcc_root>/<split>/<wav_stem>.npy
    => wav_stem == safe_id

So MFCC path is:
    <mfcc_root>/<split>/<safe_id>.npy

Returns:
    (video_tensor, mfcc_tensor, label_tensor, video_id)

Shapes:
- video_tensor: FloatTensor [N_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE]
- mfcc_tensor : FloatTensor [1, N_MFCC, T]  (variable T, or padded if mfcc_max_len is set)
- label       : FloatTensor scalar

Dependencies:
    pip install torch torchvision numpy pandas pillow

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# hashlib is used to recreate the same safe_id naming scheme
# used by the preprocessing scripts
import hashlib

# Path is used for clean file/folder handling
from pathlib import Path

# Optional is used because mfcc_max_len can be None
from typing import Optional

# NumPy is used for loading and editing MFCC arrays
import numpy as np

# pandas is used to read the split CSV files
import pandas as pd

# PyTorch is used for tensor creation and Dataset inheritance
import torch

# PIL loads image frames in RGB format
from PIL import Image

# Base Dataset class for PyTorch dataloaders
from torch.utils.data import Dataset

# torchvision transforms handle resizing, tensor conversion, and normalization
from torchvision import transforms


def make_safe_id(video_id: str) -> str:
    """
    Recreate the safe_id exactly as used in both frame and audio preprocessing.

    This is important because all stored files are named by safe_id,
    not by the raw video_id.
    """
    return hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:16]


class FakeAVCelebAVDataset(Dataset):
    """
    Multimodal Audio + Video dataset for FakeAVCeleb_v1.2.

    Each sample returns:
    - video frame tensor
    - MFCC tensor
    - binary label
    - original video_id

    This loader is built around the split CSV file so data loading stays
    aligned with the train/val/test partitions used elsewhere in VerifAI.
    """

    def __init__(
        self,
        split: str,
        n_frames: int = 5,
        image_size: int = 224,
        n_mfcc: int = 40,
        mfcc_max_len: Optional[int] = None,
        strict: bool = True,
        csv_root: str | Path = "data/splits",
        frames_root: str | Path = "data/interim/frames/FakeAVCeleb_v1.2",
        mfcc_root: str | Path = "data/processed/audio_features/FakeAVCeleb_v1.2",
    ):
        """
        Args:
            split:
                one of train | val | test

            n_frames:
                number of expected frames per sample

            image_size:
                target image resolution for all frames

            n_mfcc:
                expected number of MFCC coefficients

            mfcc_max_len:
                if set, pad or trim the MFCC time axis to a fixed length
                so batching becomes easier

            strict:
                if True, missing/corrupted data raises an error
                if False, missing inputs are replaced with zeros
        """

        # Normalize split name and validate it early
        split = str(split).lower().strip()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.split = split
        self.n_frames = int(n_frames)
        self.image_size = int(image_size)
        self.n_mfcc = int(n_mfcc)
        self.mfcc_max_len = None if mfcc_max_len is None else int(mfcc_max_len)
        self.strict = bool(strict)

        # Build path to the split CSV
        self.csv_path = Path(csv_root) / f"fakeavceleb_{split}.csv"
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        # Read split metadata
        self.df = pd.read_csv(self.csv_path)

        # Check that the columns required by this loader are present
        required_cols = {"video_id", "video_path", "label"}
        missing = required_cols.difference(self.df.columns)
        if missing:
            raise ValueError(f"{self.csv_path} missing required columns: {missing}")

        # Store root paths for this split only
        self.frames_root = Path(frames_root) / split
        self.mfcc_root = Path(mfcc_root) / split

        # Standard image preprocessing used for frame-based vision backbones
        self.frame_tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        """Return number of multimodal samples available in the split CSV."""
        return len(self.df)

    # ----------------------------
    # Frames
    # ----------------------------
    def _frame_folder(self, safe_id: str) -> Path:
        """
        Build frame folder path for a given safe_id.
        """
        return self.frames_root / safe_id

    def _load_frames_tensor(self, folder: Path) -> torch.Tensor:
        """
        Load exactly N frames from the folder using the expected naming scheme:
            frame_000.jpg ... frame_004.jpg

        If the exact preferred filenames do not all exist, fall back to any
        available frame_*.jpg files sorted alphabetically.
        """
        # Prefer exact filenames because they match the extraction script output
        preferred = [folder / f"frame_{i:03d}.jpg" for i in range(self.n_frames)]

        if all(p.exists() for p in preferred):
            paths = preferred
        else:
            # Fallback: use any available frame files
            paths = sorted(folder.glob("frame_*.jpg"))

            if len(paths) < self.n_frames:
                if self.strict:
                    raise FileNotFoundError(
                        f"Not enough frames in {folder}. Needed {self.n_frames}, found {len(paths)}"
                    )

                # In non-strict mode, return safe zero input instead of crashing
                return torch.zeros(
                    (self.n_frames, 3, self.image_size, self.image_size),
                    dtype=torch.float32
                )

            paths = paths[: self.n_frames]

        frames = []

        # Load and preprocess each frame consistently
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.frame_tf(img))

        return torch.stack(frames, dim=0)  # [N, 3, H, W]

    # ----------------------------
    # MFCC
    # ----------------------------
    def _mfcc_path(self, safe_id: str) -> Path:
        """
        Build MFCC file path for a given safe_id.
        """
        return self.mfcc_root / f"{safe_id}.npy"

    def _load_mfcc_tensor(self, mfcc_path: Path) -> torch.Tensor:
        """
        Load MFCC tensor as shape [1, n_mfcc, T].

        If mfcc_max_len is set, the time dimension is padded or trimmed
        to a fixed size so batching is possible.
        """
        if not mfcc_path.exists():
            if self.strict:
                raise FileNotFoundError(f"Missing MFCC file: {mfcc_path}")

            # In non-strict mode, return a zero tensor placeholder
            t = 1 if self.mfcc_max_len is None else self.mfcc_max_len
            return torch.zeros((1, self.n_mfcc, t), dtype=torch.float32)

        # Expected MFCC shape is [n_mfcc, T]
        mfcc = np.load(mfcc_path)

        if mfcc.ndim != 2:
            if self.strict:
                raise ValueError(f"MFCC has wrong shape {mfcc.shape} at {mfcc_path}")

            t = 1 if self.mfcc_max_len is None else self.mfcc_max_len
            return torch.zeros((1, self.n_mfcc, t), dtype=torch.float32)

        # Handle mismatch in MFCC coefficient count
        # Strict mode raises an error, non-strict mode fixes shape by pad/trim
        if mfcc.shape[0] != self.n_mfcc:
            if self.strict:
                raise ValueError(
                    f"MFCC n_mfcc mismatch: expected {self.n_mfcc}, got {mfcc.shape[0]} at {mfcc_path}"
                )

            if mfcc.shape[0] > self.n_mfcc:
                mfcc = mfcc[: self.n_mfcc, :]
            else:
                pad_rows = self.n_mfcc - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_rows), (0, 0)), mode="constant")

        # Optionally force a fixed time-axis length for stable DataLoader batching
        if self.mfcc_max_len is not None:
            T = mfcc.shape[1]

            if T > self.mfcc_max_len:
                mfcc = mfcc[:, : self.mfcc_max_len]
            elif T < self.mfcc_max_len:
                pad_cols = self.mfcc_max_len - T
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_cols)), mode="constant")

        # Convert to tensor and add channel dimension
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, n_mfcc, T]

        # Per-sample normalization improves numerical stability
        mean = x.mean()
        std = x.std().clamp(min=1e-6)
        x = (x - mean) / std

        return x

    # ----------------------------
    # Dataset return
    # ----------------------------
    def __getitem__(self, idx: int):
        """
        Return one multimodal sample.

        Output:
            video_tensor : [N_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE]
            mfcc_tensor  : [1, N_MFCC, T]
            y            : scalar float tensor
            video_id     : original video identifier
        """
        row = self.df.iloc[idx]

        video_id = str(row["video_id"])
        label = float(int(row["label"]))

        # Recreate the same safe_id used by preprocessing outputs
        safe_id = make_safe_id(video_id)

        # ---- frames ----
        folder = self._frame_folder(safe_id)

        if not folder.exists():
            if self.strict:
                raise FileNotFoundError(
                    f"Frame folder not found for safe_id={safe_id} (video_id={video_id}) -> {folder}"
                )

            # In non-strict mode, use zero frames instead of stopping execution
            video_tensor = torch.zeros(
                (self.n_frames, 3, self.image_size, self.image_size),
                dtype=torch.float32
            )
        else:
            video_tensor = self._load_frames_tensor(folder)

        # ---- MFCC ----
        mfcc_path = self._mfcc_path(safe_id)
        mfcc_tensor = self._load_mfcc_tensor(mfcc_path)

        # Store label as float tensor for binary classification compatibility
        y = torch.tensor(label, dtype=torch.float32)

        return video_tensor, mfcc_tensor, y, video_id


if __name__ == "__main__":
    # Simple sanity check to verify the loader works and tensor shapes match expectations
    ds = FakeAVCelebAVDataset("val", strict=False, mfcc_max_len=200)

    v, a, y, vid = ds[0]

    print("video:", v.shape, "mfcc:", a.shape, "label:", y.item(), "video_id:", vid)
    print("video_sum", float(v.abs().sum()), "mfcc_sum", float(a.abs().sum()), "T", a.shape[-1])