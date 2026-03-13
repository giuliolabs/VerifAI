"""
FaceForensics++ Frame Dataset Loader
====================================

This module defines a reusable PyTorch Dataset for loading frame-level
samples from the FaceForensics++ C23 dataset. It is designed to support
more than one folder structure, so the same loader can still be used
even if the extracted frames are organised differently across experiments.

Supported layouts:

(A) Class-folder layout
    root_dir/{real|fake}/{video_id}/frame_*.jpg

(B) ID-folder layout
    root_dir/{frame_id}/frame_*.jpg

For layout (B), labels are inferred from an existing mapping file.
The mapping file links each folder/frame ID back to the original source
video path, which is then used to decide whether the sample is real or fake.

Label rule:
- source path containing "/original/" => real (0)
- otherwise => fake (1)

This design keeps the loader flexible and avoids rewriting dataset logic
when the frame extraction pipeline changes.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# os is used for file and folder navigation
import os

# PIL is used to load frame images in RGB format
from PIL import Image

# PyTorch Dataset base class
from torch.utils.data import Dataset


def _load_label_map_from_mapping(mapping_file: str = None):
    """
    Load a label map from an existing mapping file.

    The mapping file is expected to provide at least:
    - frame_id / folder_id
    - original source path

    If mapping_file is not provided explicitly, the function tries
    several common file locations inside the raw dataset folder.

    Returns:
        dict mapping frame_id -> label
    """

    candidates = []

    # If the caller provides a specific mapping file, try that first
    if mapping_file:
        candidates.append(mapping_file)

    # Fallback: try a few likely filenames that may already exist
    # in the raw dataset folder
    candidates.extend([
        os.path.join("data", "raw", "FaceForensics++_C23", "frame_index.csv"),
        os.path.join("data", "raw", "FaceForensics++_C23", "mapping.csv"),
        os.path.join("data", "raw", "FaceForensics++_C23", "index.csv"),
        os.path.join("data", "raw", "FaceForensics++_C23", "frames.csv"),
        os.path.join("data", "raw", "FaceForensics++_C23", "frame_mapping.csv"),
    ])

    mapping_path = None

    # Use the first existing mapping file found
    for c in candidates:
        if os.path.exists(c):
            mapping_path = c
            break

    # Fail clearly if no mapping source is available
    if mapping_path is None:
        raise FileNotFoundError(
            "Could not find a mapping file to infer labels for your current folder layout.\n"
            "Your frames are stored as split/<id>/frame_*.jpg, so labels must come from an existing mapping.\n"
            "Pass mapping_file=... when creating FFPPFrameDataset, or place the mapping at:\n"
            "  data/raw/FaceForensics++_C23/frame_index.csv\n"
        )

    label_map = {}

    # Read the mapping file line by line.
    # This version keeps parsing simple and avoids depending on csv module
    # if the file format is already basic and consistent.
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]

            # At least 3 fields are needed:
            # frame_id, video_id, source_path
            if len(parts) < 3:
                continue

            frame_id = parts[0]
            source_path = parts[2].replace("\\", "/")

            # Label inference rule:
            # original videos are real, everything else is treated as fake
            label = 0 if "/original/" in source_path else 1
            label_map[frame_id] = label

    return label_map


class FFPPFrameDataset(Dataset):
    """
    PyTorch Dataset for loading FaceForensics++ frame samples.

    Supports two folder layouts:

    (A) With class folders:
        root_dir/{real|fake}/{video_id}/frame_*.jpg

    (B) Without class folders:
        root_dir/{frame_id}/frame_*.jpg

    In layout (B), labels are inferred using a mapping file.

    Returns:
        (image_tensor, label_int, video_id_str)

    Notes:
    - For layout (A), video_id_str is the real video folder name
    - For layout (B), video_id_str is the folder/frame ID
    """

    def __init__(self, root_dir: str, transform=None, mapping_file: str = None):
        self.root_dir = root_dir
        self.transform = transform

        # Internal storage format:
        # (image_path, label, video_id)
        self.items = []

        # First try the simpler/classic layout with explicit real/fake folders
        used_layout_a = self._try_load_real_fake_layout()

        # If no real/fake structure was found, assume ID-folder layout
        # and infer labels using the mapping file
        if not used_layout_a:
            label_map = _load_label_map_from_mapping(mapping_file)
            self._load_id_folder_layout(label_map)

    def _try_load_real_fake_layout(self) -> bool:
        """
        Try to load the dataset using the layout:

            root_dir/{real|fake}/{video_id}/frame_*.jpg

        Returns:
            True if at least one valid image was found
            False otherwise
        """
        found_any = False

        # The folder name itself provides the class label here
        for class_name, label in [("real", 0), ("fake", 1)]:
            class_dir = os.path.join(self.root_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            for video_id in sorted(os.listdir(class_dir)):
                video_dir = os.path.join(class_dir, video_id)

                if not os.path.isdir(video_dir):
                    continue

                for file_name in sorted(os.listdir(video_dir)):
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(video_dir, file_name)
                        self.items.append((img_path, label, video_id))
                        found_any = True

        return found_any

    def _load_id_folder_layout(self, label_map):
        """
        Load the dataset using the layout:

            root_dir/{frame_id}/frame_*.jpg

        Only folders that exist in the label_map are included.
        This avoids loading frames whose labels cannot be determined.
        """
        for folder_name in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            # Skip folders that cannot be matched to a known label
            if folder_name not in label_map:
                continue

            label = int(label_map[folder_name])

            for file_name in sorted(os.listdir(folder_path)):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder_path, file_name)
                    self.items.append((img_path, label, folder_name))

    def __len__(self):
        """
        Return total number of frame samples available.
        """
        return len(self.items)

    def __getitem__(self, index):
        """
        Load one frame sample.

        Returns:
            image: transformed image tensor (or raw PIL image if no transform)
            label: integer class label
            video_id: associated video/folder identifier
        """
        img_path, label, video_id = self.items[index]

        # Always load as RGB so image shape stays consistent
        image = Image.open(img_path).convert("RGB")

        # Apply preprocessing / augmentation if provided
        if self.transform is not None:
            image = self.transform(image)

        return image, label, video_id