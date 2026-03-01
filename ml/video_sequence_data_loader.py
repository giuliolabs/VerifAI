"""
Video sequence dataloader for FaceForensics++ C23 extracted frames (hashed folders).

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
  video_path,label,video_id   (video_path used for matching)

Returns:
  x: torch.FloatTensor [T, 3, H, W]
  y: torch.FloatTensor scalar
  safe_id: str
"""

from __future__ import annotations

from pathlib import Path
import csv
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FFPPSequenceDataset(Dataset):
    def __init__(
        self,
        frames_root: str | Path,
        split: str,
        split_csv: str | Path,
        img_size: int = 224,
        t: int = 5,
        train: bool = True,
    ):
        self.frames_root = Path(frames_root)
        self.split = split
        self.split_csv = Path(split_csv)
        self.split_dir = self.frames_root / split
        self.t = int(t)

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {self.split_dir}")

        id_map = self.split_dir / "_id_map.csv"
        if not id_map.exists():
            raise FileNotFoundError(f"Missing id map: {id_map}")

        self.transform = self._make_transforms(img_size=img_size, train=train)

        vpath_to_label = self._read_split_labels(self.split_csv)
        safe_to_vpath = self._read_id_map(id_map)

        # samples: (safe_id, label)
        self.samples: List[Tuple[str, int]] = []
        for safe_id, vpath in safe_to_vpath.items():
            if vpath not in vpath_to_label:
                continue
            vid_folder = self.split_dir / safe_id
            if not vid_folder.exists():
                continue

            # at least 1 frame exists
            if len(list(vid_folder.glob("frame_*.jpg"))) == 0:
                continue

            self.samples.append((safe_id, int(vpath_to_label[vpath])))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No usable sequences found for split={split}. "
                f"Check frames and that _id_map.csv matches {self.split_csv}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        safe_id, label = self.samples[idx]
        vid_folder = self.split_dir / safe_id

        # Prefer exact frame_000..frame_004 for stability
        preferred = [vid_folder / f"frame_{i:03d}.jpg" for i in range(self.t)]
        if all(p.exists() for p in preferred):
            paths = preferred
        else:
            paths = sorted(vid_folder.glob("frame_*.jpg"))
            if not paths:
                raise FileNotFoundError(f"No frames found in {vid_folder}")
            # take first T, pad by repeating last
            paths = paths[: self.t]
            while len(paths) < self.t:
                paths.append(paths[-1])

        frames: List[torch.Tensor] = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.transform(img))   # [3,H,W]

        x = torch.stack(frames, dim=0)          # [T,3,H,W]
        y = torch.tensor(label, dtype=torch.float32)
        return x, y, safe_id

    @staticmethod
    def _make_transforms(img_size: int, train: bool):
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
        safe_to_vpath: dict[str, str] = {}
        with id_map_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                safe_id = str(row.get("safe_id", "")).strip()
                vpath = str(row.get("video_path", "")).replace("\\", "/").strip()
                if safe_id and vpath:
                    safe_to_vpath[safe_id] = vpath
        return safe_to_vpath
