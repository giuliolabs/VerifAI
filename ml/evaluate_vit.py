"""
Evaluate ViT Baseline on FaceForensics++ C23 Frames (Hashed Folders)
====================================================================

This script evaluates the trained ViT visual baseline on the
FaceForensics++ C23 test split. It loads frame-level samples from
hashed SAFE_ID folders, restores the best trained checkpoint, and
computes consistent binary classification metrics for later comparison
and consolidation.

Writes consistent metrics for consolidation:
- Accuracy
- Balanced Accuracy
- F1 (from sklearn report)
- ROC-AUC
- Average Precision (AP)
- MCC
- Confusion matrix + classification report

Saves:
    experiments/results/ffpp_c23_vit_baseline/test_report.txt
    experiments/results/ffpp_c23_vit_baseline/test_metrics.json

Run:
    python -m ml.evaluate_vit

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Add project root to Python path so internal modules can be imported
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# csv is used to read split labels and SAFE_ID mapping files
import csv

# NumPy is used for readable metric printing
import numpy as np

# PyTorch core utilities
import torch
from torch.utils.data import Dataset, DataLoader

# torchvision transforms prepare images for ViT input
from torchvision import transforms

# PIL is used to load image frames in RGB format
from PIL import Image

# tqdm provides progress bars during evaluation
from tqdm import tqdm

# Project model builder and shared metrics utilities
from ml.models.video.vit import build_vit_binary
from ml.metrics import compute_binary_metrics, save_metrics_report


# Root folder containing extracted frame folders
FRAMES_ROOT = Path("data/interim/frames/FaceForensics++_C23")

# Folder containing split CSV files
SPLITS_DIR = Path("data/splits")
TEST_CSV = SPLITS_DIR / "faceforensics++_c23_test.csv"

# Output folder and checkpoint path for the trained ViT baseline
OUT_DIR = Path("experiments/results/ffpp_c23_vit_baseline")
CKPT_PATH = OUT_DIR / "best_model.pt"

# ViT input size and evaluation batch settings
IMG_SIZE = 224
BATCH = 32
NUM_WORKERS = 0


def _torch_load_compat(path: Path, device: str):
    """
    Load PyTorch checkpoint in a version-compatible way.

    Some torch versions support weights_only=False explicitly,
    while older versions do not.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_transforms() -> transforms.Compose:
    """
    Build deterministic preprocessing pipeline for test-time evaluation.

    ViT expects 224x224 input and ImageNet-style normalization.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def read_split_labels(split_csv: Path) -> dict[str, int]:
    """
    Read mapping from original video_path -> label from the split CSV.

    This mapping is later joined with SAFE_ID folders through _id_map.csv.
    """
    mapping: dict[str, int] = {}
    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vp = str(row.get("video_path", "")).replace("\\", "/")
            if not vp:
                continue
            mapping[vp] = int(row["label"])
    return mapping


def read_id_map(id_map_path: Path) -> dict[str, str]:
    """
    Read mapping from safe_id -> original video_path.

    This allows hashed frame folders to be matched back to their
    correct labels using the split CSV.
    """
    safe_to_vpath: dict[str, str] = {}
    with id_map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            safe_id = str(row["safe_id"]).strip()
            vpath = str(row["video_path"]).replace("\\", "/").strip()
            if safe_id and vpath:
                safe_to_vpath[safe_id] = vpath
    return safe_to_vpath


class FFPPHashedFrameDataset(Dataset):
    """
    Dataset for loading frame-level samples from hashed SAFE_ID folders.

    Labels are matched by joining:
    - safe_id -> video_path from _id_map.csv
    - video_path -> label from the split CSV
    """

    def __init__(self, split: str, split_csv: Path, transform: transforms.Compose):
        self.transform = transform

        split_dir = FRAMES_ROOT / split
        id_map = split_dir / "_id_map.csv"

        if not id_map.exists():
            raise FileNotFoundError(f"Missing id map: {id_map}")

        vpath_to_label = read_split_labels(split_csv)
        safe_to_vpath = read_id_map(id_map)

        self.samples: list[tuple[Path, int]] = []

        # Build sample list as (image_path, label)
        for safe_id, vpath in safe_to_vpath.items():
            if vpath not in vpath_to_label:
                continue

            label = vpath_to_label[vpath]
            vid_folder = split_dir / safe_id
            if not vid_folder.exists():
                continue

            for img_path in sorted(vid_folder.glob("frame_*.jpg")):
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No frames found for split={split} under {split_dir}")

    def __len__(self):
        """Return total number of frame samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load one frame sample and return:
        - transformed image tensor
        - label tensor
        - image path for traceability/debugging
        """
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y, str(img_path)


@torch.no_grad()
def main() -> None:
    """
    Main evaluation pipeline for the ViT frame baseline.

    This function:
    - loads the test dataset
    - restores the trained ViT checkpoint
    - runs frame-level inference
    - computes shared binary metrics
    - saves TXT and JSON evaluation outputs
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Build test dataset and loader
    ds = FFPPHashedFrameDataset("test", TEST_CSV, transform=make_transforms())
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    # Rebuild model architecture and load trained weights
    model = build_vit_binary(pretrained=False, img_size=IMG_SIZE).to(device)
    ckpt = _torch_load_compat(CKPT_PATH, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true: list[int] = []
    y_score: list[float] = []

    # Run inference over all test frames
    for x, y, _ in tqdm(loader, desc="Evaluating"):
        x = x.to(device)

        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

        y_true.extend([int(v) for v in y.cpu().numpy().tolist()])
        y_score.extend([float(p) for p in probs.tolist()])

    # Compute consistent binary classification metrics
    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        include_curves=True,
    )

    print("\nVIT RESULTS (FaceForensics++ C23 frame baseline)")
    print("Checkpoint:", CKPT_PATH)
    print("Num samples:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("Balanced accuracy:", getattr(metrics, "balanced_accuracy", None))
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("MCC:", getattr(metrics, "mcc", None))
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)

    # Save both human-readable and machine-readable outputs
    txt_path, json_path = save_metrics_report(
        out_dir=OUT_DIR,
        name="test",
        metrics=metrics,
        extra={
            "dataset": "FaceForensics++ C23",
            "model": "ViT frame baseline",
            "img_size": IMG_SIZE,
            "threshold": 0.5,
            "checkpoint": str(CKPT_PATH),
        },
    )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()