"""
MFCC Dataset loader (loads .npy MFCC features using split CSV).

Expected:
- split CSV has columns: video_id,label (and optionally video_path,av_category)
- MFCC files stored as: <mfcc_root>/<video_id>.npy

Dependencies:
    pip install numpy pandas torch

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FIXED_T = 300  # number of MFCC time-steps (pad/crop)

class MFCCDataset(Dataset):
    def __init__(self, split_csv: str, mfcc_root: str):
        self.df = pd.read_csv(split_csv)
        self.mfcc_root = Path(mfcc_root)

        if "video_id" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain at least: video_id,label")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import hashlib
        from pathlib import Path

        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        label = int(row["label"])

        video_path_str = str(row.get("video_path", ""))
        video_name = Path(video_path_str).name if video_path_str else ""
        video_stem = Path(video_path_str).stem if video_path_str else ""

        # Build candidate MFCC filenames (to handle different naming conventions)
        candidates = [f"{video_id}.npy"]

        # 1) CSV video_id as-is

        # 2) If video_id includes extension like ".mp4", try without it
        if video_id.lower().endswith(".mp4"):
            candidates.append(f"{Path(video_id).stem}.npy")

        # 3) Try using the original video filename/stem from video_path
        if video_name:
            candidates.append(f"{video_name}.npy")  # e.g. something.mp4.npy (rare but possible)
        if video_stem:
            candidates.append(f"{video_stem}.npy")  # e.g. something.npy

        # 4) Try hashed IDs (common in pipelines)
        if video_path_str:
            candidates.append(hashlib.md5(video_path_str.encode("utf-8")).hexdigest() + ".npy")
            candidates.append(hashlib.md5(video_path_str.replace("\\", "/").encode("utf-8")).hexdigest() + ".npy")
        if video_name:
            candidates.append(hashlib.md5(video_name.encode("utf-8")).hexdigest() + ".npy")
        candidates.append(hashlib.md5(video_id.encode("utf-8")).hexdigest() + ".npy")

        # 5) Truncated hash naming (your MFCC files are 16 hex chars)
        if video_path_str:
            candidates.append(hashlib.md5(video_path_str.encode("utf-8")).hexdigest()[:16] + ".npy")
            candidates.append(hashlib.md5(video_path_str.replace("\\", "/").encode("utf-8")).hexdigest()[:16] + ".npy")
        if video_name:
            candidates.append(hashlib.md5(video_name.encode("utf-8")).hexdigest()[:16] + ".npy")
        candidates.append(hashlib.md5(video_id.encode("utf-8")).hexdigest()[:16] + ".npy")

        # 6) Truncated SHA1 / SHA256 naming (16 hex chars) - common alternative
        if video_path_str:
            norm_path = video_path_str.replace("\\", "/")

            candidates.append(hashlib.sha1(video_path_str.encode("utf-8")).hexdigest()[:16] + ".npy")
            candidates.append(hashlib.sha1(norm_path.encode("utf-8")).hexdigest()[:16] + ".npy")

            candidates.append(hashlib.sha256(video_path_str.encode("utf-8")).hexdigest()[:16] + ".npy")
            candidates.append(hashlib.sha256(norm_path.encode("utf-8")).hexdigest()[:16] + ".npy")

        if video_name:
            candidates.append(hashlib.sha1(video_name.encode("utf-8")).hexdigest()[:16] + ".npy")
            candidates.append(hashlib.sha256(video_name.encode("utf-8")).hexdigest()[:16] + ".npy")

        candidates.append(hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:16] + ".npy")
        candidates.append(hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:16] + ".npy")

        # Pick the first existing candidate
        mfcc_path = None
        for fname in candidates:
            p = self.mfcc_root / fname
            if p.exists():
                mfcc_path = p
                break

        if mfcc_path is None:
            raise FileNotFoundError(
                "Missing MFCC file. Tried:\n  - " + "\n  - ".join(str(self.mfcc_root / c) for c in candidates[:8])
            )

        mfcc = np.load(mfcc_path)  # shape: (n_mfcc, T)

        # Ensure consistent time length for batching
        T = mfcc.shape[1]
        if T < FIXED_T:
            pad_width = FIXED_T - T
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        elif T > FIXED_T:
            start = (T - FIXED_T) // 2
            mfcc = mfcc[:, start:start + FIXED_T]

        # Convert to torch tensor: [1, n_mfcc, FIXED_T]
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        # Per-sample normalization (stable)
        mean = x.mean()
        std = x.std().clamp(min=1e-6)
        x = (x - mean) / std

        y = torch.tensor(label, dtype=torch.float32)
        return x, y, video_id

