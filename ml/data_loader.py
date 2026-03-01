import os
from PIL import Image
from torch.utils.data import Dataset


def _load_label_map_from_mapping(mapping_file: str = None):
    """
    Loads frame_id -> label from an existing mapping file.
    If mapping_file is None, tries common locations.
    """
    candidates = []
    if mapping_file:
        candidates.append(mapping_file)

    # Try a couple of likely places (no new files needed)
    candidates.extend([
        os.path.join("data", "raw", "FaceForensics++_C23", "frame_index.csv"),
        os.path.join("data", "raw", "FaceForensics++_C23", "mapping.csv"),
        os.path.join("data", "raw", "FaceForensics++_C23", "index.csv"),
        os.path.join("data", "raw", "FaceForensics++_C23", "frames.csv"),
        os.path.join("data", "raw", "FaceForensics++_C23", "frame_mapping.csv"),
    ])

    mapping_path = None
    for c in candidates:
        if os.path.exists(c):
            mapping_path = c
            break

    if mapping_path is None:
        raise FileNotFoundError(
            "Could not find a mapping file to infer labels for your current folder layout.\n"
            "Your frames are stored as split/<id>/frame_*.jpg, so labels must come from an existing mapping.\n"
            "Pass mapping_file=... when creating FFPPFrameDataset, or place the mapping at:\n"
            "  data/raw/FaceForensics++_C23/frame_index.csv\n"
        )

    label_map = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue

            frame_id = parts[0]
            source_path = parts[2].replace("\\", "/")

            # label rule
            label = 0 if "/original/" in source_path else 1
            label_map[frame_id] = label

    return label_map


class FFPPFrameDataset(Dataset):
    """
    Supports two folder layouts:

    (A) With class folders:
        root_dir/{real|fake}/{video_id}/frame_*.jpg

    (B) Without class folders (your current layout):
        root_dir/{frame_id}/frame_*.jpg

        In this case, labels are inferred from a mapping file that contains lines like:
            <frame_id>,<video_id>,data/raw/FaceForensics++_C23/original/351.mp4
            <frame_id>,<video_id>,data/raw/FaceForensics++_C23/Face2Face/605_591.mp4

        Rule:
            source_path containing "/original/" => label 0 (real)
            otherwise => label 1 (fake)

    Returns: (image_tensor, label_int, video_id_str)
    For layout (B), video_id_str will be the folder name (frame_id).
    """

    def __init__(self, root_dir: str, transform=None, mapping_file: str = None):
        self.root_dir = root_dir
        self.transform = transform
        self.items = []  # (img_path, label, video_id)

        # 1) Try layout A (real/fake)
        used_layout_a = self._try_load_real_fake_layout()

        # 2) If layout A didn't work, try layout B (folders are IDs)
        if not used_layout_a:
            label_map = _load_label_map_from_mapping(mapping_file)
            self._load_id_folder_layout(label_map)

    def _try_load_real_fake_layout(self) -> bool:
        found_any = False
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
        root_dir/{frame_id}/frame_*.jpg
        """
        for folder_name in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # only include folders we can label
            if folder_name not in label_map:
                continue

            label = int(label_map[folder_name])

            for file_name in sorted(os.listdir(folder_path)):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder_path, file_name)
                    self.items.append((img_path, label, folder_name))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, label, video_id = self.items[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, video_id
