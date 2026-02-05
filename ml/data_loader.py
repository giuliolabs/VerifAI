"""
FF++ Frame Dataset Loader (for baseline CNN)
Dependencies:
    pip install torch torchvision pillow
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class FFPPFrameDataset(Dataset):
    """
    Expects folder structure:
      data/interim/frames/FaceForensics++_C23/{split}/{real|fake}/{video_id}/frame_*.jpg
    Returns: (image_tensor, label_int, video_id_str)
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.items = []  # (img_path, label, video_id)

        for class_name, label in [("real", 0), ("fake", 1)]:
            class_dir = os.path.join(root_dir, class_name)
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, label, video_id = self.items[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, video_id
