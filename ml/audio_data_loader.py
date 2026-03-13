"""
MFCC Dataset Loader (Loads .npy MFCC Features Using Split CSV)
==============================================================

This module defines the dataset loader used for the audio-only experiments
in the VerifAI framework. It loads MFCC feature files stored as NumPy arrays
(.npy) and matches them with labels defined in a split CSV file.

Expected:
- The split CSV must contain at least the columns:
      video_id, label
  and may optionally include:
      video_path, av_category

- MFCC files should be stored under:
      <mfcc_root>/<video_id>.npy

Because preprocessing pipelines sometimes rename files (e.g., hashing
video identifiers), this loader attempts multiple candidate filenames
to locate the correct MFCC file.

Dependencies:
    pip install numpy pandas torch

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from pathlib import Path

# NumPy loads MFCC feature arrays
import numpy as np

# pandas reads the split CSV files
import pandas as pd

# PyTorch provides tensor conversion and Dataset base class
import torch
from torch.utils.data import Dataset


# Fixed number of MFCC time steps used for batching.
# MFCC sequences shorter than this are padded, longer ones are cropped.
FIXED_T = 300


class MFCCDataset(Dataset):
    """
    PyTorch Dataset for loading MFCC features stored as .npy files.

    Each sample returns:
        x : Tensor [1, n_mfcc, FIXED_T]
        y : Tensor scalar label
        video_id : identifier string

    The dataset uses a split CSV to determine which samples belong to
    train / validation / test sets.
    """

    def __init__(self, split_csv: str, mfcc_root: str):
        """
        Args:
            split_csv:
                path to the CSV file defining the dataset split

            mfcc_root:
                directory containing MFCC .npy files
        """

        # Load split metadata
        self.df = pd.read_csv(split_csv)

        # Root folder containing MFCC feature files
        self.mfcc_root = Path(mfcc_root)

        # Validate that required CSV columns exist
        if "video_id" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain at least: video_id,label")

    def __len__(self):
        """
        Return total number of samples in the split CSV.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Load one MFCC sample and return its feature tensor and label.
        """

        import hashlib
        from pathlib import Path

        # Read metadata row
        row = self.df.iloc[idx]

        video_id = str(row["video_id"])
        label = int(row["label"])

        # Some pipelines store additional information about the original video
        video_path_str = str(row.get("video_path", ""))

        # Extract useful components from the path if available
        video_name = Path(video_path_str).name if video_path_str else ""
        video_stem = Path(video_path_str).stem if video_path_str else ""

        # -------------------------------------------------------
        # Candidate MFCC filenames
        # -------------------------------------------------------
        # Because preprocessing pipelines may rename files, we
        # attempt several possible filename formats.

        candidates = [f"{video_id}.npy"]

        # If video_id includes extension like ".mp4", try removing it
        if video_id.lower().endswith(".mp4"):
            candidates.append(f"{Path(video_id).stem}.npy")

        # Try using the filename from video_path
        if video_name:
            candidates.append(f"{video_name}.npy")

        # Try using the stem of video_path
        if video_stem:
            candidates.append(f"{video_stem}.npy")

        # -------------------------------------------------------
        # Hashed filename candidates
        # -------------------------------------------------------
        if video_path_str:
            candidates.append(hashlib.md5(video_path_str.encode("utf-8")).hexdigest() + ".npy")
            candidates.append(hashlib.md5(video_path_str.replace("\\", "/").encode("utf-8")).hexdigest() + ".npy")

        if video_name:
            candidates.append(hashlib.md5(video_name.encode("utf-8")).hexdigest() + ".npy")

        candidates.append(hashlib.md5(video_id.encode("utf-8")).hexdigest() + ".npy")

        # -------------------------------------------------------
        # Truncated MD5 hashes (16 chars)
        # -------------------------------------------------------
        if video_path_str:
            candidates.append(hashlib.md5(video_path_str.encode("utf-8")).hexdigest()[:16] + ".npy")
            candidates.append(hashlib.md5(video_path_str.replace("\\", "/").encode("utf-8")).hexdigest()[:16] + ".npy")

        if video_name:
            candidates.append(hashlib.md5(video_name.encode("utf-8")).hexdigest()[:16] + ".npy")

        candidates.append(hashlib.md5(video_id.encode("utf-8")).hexdigest()[:16] + ".npy")

        # -------------------------------------------------------
        # SHA1 / SHA256 hash naming patterns
        # -------------------------------------------------------
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

        # -------------------------------------------------------
        # Locate the first existing MFCC file
        # -------------------------------------------------------
        mfcc_path = None

        for fname in candidates:
            p = self.mfcc_root / fname
            if p.exists():
                mfcc_path = p
                break

        # If no candidate matched, raise an informative error
        if mfcc_path is None:
            raise FileNotFoundError(
                "Missing MFCC file. Tried:\n  - "
                + "\n  - ".join(str(self.mfcc_root / c) for c in candidates[:8])
            )

        # -------------------------------------------------------
        # Load MFCC feature array
        # -------------------------------------------------------
        mfcc = np.load(mfcc_path)  # expected shape: (n_mfcc, T)

        # -------------------------------------------------------
        # Ensure consistent time length for batching
        # -------------------------------------------------------
        T = mfcc.shape[1]

        if T < FIXED_T:
            pad_width = FIXED_T - T
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")

        elif T > FIXED_T:
            # Crop central segment for consistency
            start = (T - FIXED_T) // 2
            mfcc = mfcc[:, start:start + FIXED_T]

        # Convert to PyTorch tensor and add channel dimension
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        # -------------------------------------------------------
        # Per-sample normalization (improves training stability)
        # -------------------------------------------------------
        mean = x.mean()
        std = x.std().clamp(min=1e-6)
        x = (x - mean) / std

        # Binary label tensor
        y = torch.tensor(label, dtype=torch.float32)

        return x, y, video_id