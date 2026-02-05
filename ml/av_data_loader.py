"""
Multimodal (Audio+Video) dataset loader for FakeAVCeleb_v1.2 (v1).

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
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_safe_id(video_id: str) -> str:
    """Must match pipelines/video and pipelines/audio."""
    return hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:16]


class FakeAVCelebAVDataset(Dataset):
    """
    Audio+Video dataset for FakeAVCeleb_v1.2 based on split CSV.
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
            split: train | val | test
            n_frames: number of frames expected per video (default 5)
            image_size: resize to (image_size, image_size)
            n_mfcc: MFCC coefficient count (default 40)
            mfcc_max_len: if set, pad/trim time axis to fixed length (recommended for batching)
            strict: if True -> raise when frames/mfcc missing; if False -> return zeros
        """
        split = str(split).lower().strip()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.split = split
        self.n_frames = int(n_frames)
        self.image_size = int(image_size)
        self.n_mfcc = int(n_mfcc)
        self.mfcc_max_len = None if mfcc_max_len is None else int(mfcc_max_len)
        self.strict = bool(strict)

        self.csv_path = Path(csv_root) / f"fakeavceleb_{split}.csv"
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        required_cols = {"video_id", "video_path", "label"}
        missing = required_cols.difference(self.df.columns)
        if missing:
            raise ValueError(f"{self.csv_path} missing required columns: {missing}")

        self.frames_root = Path(frames_root) / split
        self.mfcc_root = Path(mfcc_root) / split

        self.frame_tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    # ----------------------------
    # Frames
    # ----------------------------
    def _frame_folder(self, safe_id: str) -> Path:
        return self.frames_root / safe_id

    def _load_frames_tensor(self, folder: Path) -> torch.Tensor:
        """
        Load exactly N frames from folder using YOUR naming scheme:
            frame_000.jpg ... frame_004.jpg
        """
        # preferred exact names based on your extractor (0..N-1)
        preferred = [folder / f"frame_{i:03d}.jpg" for i in range(self.n_frames)]
        if all(p.exists() for p in preferred):
            paths = preferred
        else:
            # fallback: any frame_*.jpg sorted
            paths = sorted(folder.glob("frame_*.jpg"))
            if len(paths) < self.n_frames:
                if self.strict:
                    raise FileNotFoundError(
                        f"Not enough frames in {folder}. Needed {self.n_frames}, found {len(paths)}"
                    )
                return torch.zeros((self.n_frames, 3, self.image_size, self.image_size), dtype=torch.float32)
            paths = paths[: self.n_frames]

        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.frame_tf(img))
        return torch.stack(frames, dim=0)  # [N, 3, H, W]

    # ----------------------------
    # MFCC
    # ----------------------------
    def _mfcc_path(self, safe_id: str) -> Path:
        return self.mfcc_root / f"{safe_id}.npy"

    def _load_mfcc_tensor(self, mfcc_path: Path) -> torch.Tensor:
        """
        Load MFCC as [1, n_mfcc, T] with optional pad/trim to mfcc_max_len.
        """
        if not mfcc_path.exists():
            if self.strict:
                raise FileNotFoundError(f"Missing MFCC file: {mfcc_path}")
            # safe zeros
            t = 1 if self.mfcc_max_len is None else self.mfcc_max_len
            return torch.zeros((1, self.n_mfcc, t), dtype=torch.float32)

        mfcc = np.load(mfcc_path)  # expected (n_mfcc, T)
        if mfcc.ndim != 2:
            if self.strict:
                raise ValueError(f"MFCC has wrong shape {mfcc.shape} at {mfcc_path}")
            t = 1 if self.mfcc_max_len is None else self.mfcc_max_len
            return torch.zeros((1, self.n_mfcc, t), dtype=torch.float32)

        # Fix n_mfcc mismatch (strict -> error; non-strict -> pad/trim)
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

        # Optional fixed-length time axis for DataLoader stacking
        if self.mfcc_max_len is not None:
            T = mfcc.shape[1]
            if T > self.mfcc_max_len:
                mfcc = mfcc[:, : self.mfcc_max_len]
            elif T < self.mfcc_max_len:
                pad_cols = self.mfcc_max_len - T
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_cols)), mode="constant")

        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, n_mfcc, T]

        # Per-sample normalization (stable)
        mean = x.mean()
        std = x.std().clamp(min=1e-6)
        x = (x - mean) / std
        return x

    # ----------------------------
    # Dataset return
    # ----------------------------
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        label = float(int(row["label"]))

        safe_id = make_safe_id(video_id)

        # ---- frames ----
        folder = self._frame_folder(safe_id)
        if not folder.exists():
            if self.strict:
                raise FileNotFoundError(
                    f"Frame folder not found for safe_id={safe_id} (video_id={video_id}) -> {folder}"
                )
            video_tensor = torch.zeros((self.n_frames, 3, self.image_size, self.image_size), dtype=torch.float32)
        else:
            video_tensor = self._load_frames_tensor(folder)

        # ---- mfcc ----
        mfcc_path = self._mfcc_path(safe_id)
        mfcc_tensor = self._load_mfcc_tensor(mfcc_path)

        y = torch.tensor(label, dtype=torch.float32)
        return video_tensor, mfcc_tensor, y, video_id


if __name__ == "__main__":
    # Quick sanity test
    ds = FakeAVCelebAVDataset("val", strict=False, mfcc_max_len=200)
    v, a, y, vid = ds[0]
    print("video:", v.shape, "mfcc:", a.shape, "label:", y.item(), "video_id:", vid)
    print("video_sum", float(v.abs().sum()), "mfcc_sum", float(a.abs().sum()), "T", a.shape[-1])
