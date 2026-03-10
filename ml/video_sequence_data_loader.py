"""
Video Sequence Dataloader for FaceForensics++ C23 Extracted Frames
==================================================================

This module defines a PyTorch dataset for loading short frame sequences
from the extracted FaceForensics++ C23 dataset. Each sample represents
one video as a fixed-length sequence of frames stored inside a hashed
folder identified by a safe_id.

Expected folder structure:
data/interim/frames/FaceForensics++_C23/
  train/<safe_id>/frame_000.jpg ... frame_004.jpg
  train/_id_map.csv
  val/<safe_id>/frame_000.jpg ... frame_004.jpg
  val/_id_map.csv
  test/<safe_id>/frame_000.jpg ... frame_004.jpg
  test/_id_map.csv

Split CSV:
data/splits/faceforensics++_c23_<split>.csv with columns:
  video_path, label, video_id

Returns:
  x: torch.FloatTensor [T, 3, H, W]
  y: torch.FloatTensor scalar
  safe_id: str

This dataset is designed for sequence-based visual models such as
temporal transformers, where multiple frames from the same video are
needed rather than a single image.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Path is used for safe and readable file/folder handling
from pathlib import Path

# csv is used to read split labels and safe_id mappings
import csv

# Type hints improve readability for lists and tuples
from typing import List, Tuple

# PyTorch dataset base class and tensor utilities
import torch
from torch.utils.data import Dataset

# torchvision transforms handle resizing, tensor conversion, and normalization
from torchvision import transforms

# PIL is used to load image frames in RGB format
from PIL import Image


class FFPPSequenceDataset(Dataset):
    """
    Dataset for loading fixed-length frame sequences from FaceForensics++.

    Each sample corresponds to one video folder and returns:
    - a tensor of T frames
    - the binary label
    - the safe_id used for tracking
    """

    def __init__(
        self,
        frames_root: str | Path,
        split: str,
        split_csv: str | Path,
        img_size: int = 224,
        t: int = 5,
        train: bool = True,
    ):
        # Store main dataset settings
        self.frames_root = Path(frames_root)
        self.split = split
        self.split_csv = Path(split_csv)
        self.split_dir = self.frames_root / split
        self.t = int(t)

        # Basic validation of required folders/files
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {self.split_dir}")

        id_map = self.split_dir / "_id_map.csv"
        if not id_map.exists():
            raise FileNotFoundError(f"Missing id map: {id_map}")

        # Build transforms depending on whether the dataset is used for training
        # or evaluation. Training includes light augmentation.
        self.transform = self._make_transforms(img_size=img_size, train=train)

        # Read label mapping from split CSV and safe_id mapping from _id_map.csv
        vpath_to_label = self._read_split_labels(self.split_csv)
        safe_to_vpath = self._read_id_map(id_map)

        # Build final sample list as pairs of (safe_id, label)
        self.samples: List[Tuple[str, int]] = []
        for safe_id, vpath in safe_to_vpath.items():
            if vpath not in vpath_to_label:
                continue

            vid_folder = self.split_dir / safe_id
            if not vid_folder.exists():
                continue

            # Only keep samples that actually contain at least one frame
            if len(list(vid_folder.glob("frame_*.jpg"))) == 0:
                continue

            self.samples.append((safe_id, int(vpath_to_label[vpath])))

        # Fail early if nothing usable was found
        # This makes dataset issues easier to debug.
        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No usable sequences found for split={split}. "
                f"Check frames and that _id_map.csv matches {self.split_csv}."
            )

    def __len__(self) -> int:
        """Return number of available video sequences."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Load one video sequence.

        Returns:
            x: [T, 3, H, W]
            y: scalar float label
            safe_id: folder identifier
        """
        safe_id, label = self.samples[idx]
        vid_folder = self.split_dir / safe_id

        # Prefer exact frame_000 to frame_004 for stability and reproducibility.
        # If those exact frames are missing, fall back to available frames.
        preferred = [vid_folder / f"frame_{i:03d}.jpg" for i in range(self.t)]
        if all(p.exists() for p in preferred):
            paths = preferred
        else:
            paths = sorted(vid_folder.glob("frame_*.jpg"))
            if not paths:
                raise FileNotFoundError(f"No frames found in {vid_folder}")

            # Use first T available frames.
            # If fewer than T exist, repeat the last frame to keep fixed length.
            paths = paths[: self.t]
            while len(paths) < self.t:
                paths.append(paths[-1])

        frames: List[torch.Tensor] = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.transform(img))   # [3, H, W]

        # Stack frames into a single sequence tensor
        x = torch.stack(frames, dim=0)          # [T, 3, H, W]

        # Store label as float tensor for binary classification pipelines
        y = torch.tensor(label, dtype=torch.float32)

        return x, y, safe_id

    @staticmethod
    def _make_transforms(img_size: int, train: bool):
        """
        Build image preprocessing pipeline.

        Training mode includes light augmentation,
        while test mode uses deterministic transforms only.
        """
        if train:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _read_split_labels(split_csv: Path) -> dict[str, int]:
        """
        Read mapping from original video_path to label.

        video_path is used as the join key between the split CSV
        and the hashed frame folders via _id_map.csv.
        """
        mapping: dict[str, int] = {}
        with split_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vp = str(row.get("video_path", "")).replace("\\", "/").strip()
                if vp:
                    mapping[vp] = int(row["label"])
        return mapping

    @staticmethod
    def _read_id_map(id_map_path: Path) -> dict[str, str]:
        """
        Read mapping from safe_id folder name to original video_path.

        This allows anonymous extracted frame folders to be matched
        back to their correct labels.
        """
        safe_to_vpath: dict[str, str] = {}
        with id_map_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                safe_id = str(row.get("safe_id", "")).strip()
                vpath = str(row.get("video_path", "")).replace("\\", "/").strip()
                if safe_id and vpath:
                    safe_to_vpath[safe_id] = vpath
        return safe_to_vpath