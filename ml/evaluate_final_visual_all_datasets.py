"""
Evaluate Final Visual Backbone on All Frame Datasets
===================================================

This script evaluates the final Xception visual backbone trained across
all available frame datasets used in VerifAI. Instead of testing on a
single dataset, it combines test samples from multiple sources and
reports both:

- overall performance across all datasets
- per-dataset performance for more detailed analysis

Run:
    python -m ml.evaluate_final_visual_all_datasets

Outputs:
    experiments/results/final_visual_all_datasets_xception/test_report.txt
    experiments/results/final_visual_all_datasets_xception/test_metrics.json

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# Add project root to Python path so internal modules can be imported correctly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# csv is used to read split labels and SAFE_ID mapping files
import csv

# JSON is used to save structured evaluation results
import json

# defaultdict helps collect frame probabilities per video
from collections import defaultdict

# dataclass keeps dataset path configuration clean and readable
from dataclasses import dataclass

# NumPy is used for averaging frame probabilities and printing results
import numpy as np

# PyTorch core utilities
import torch
from torch.utils.data import Dataset, DataLoader

# torchvision transforms prepare frames for Xception input
from torchvision import transforms

# PIL loads image frames in RGB format
from PIL import Image

# tqdm provides progress bars during evaluation
from tqdm import tqdm

# sklearn metrics are used to compute a full binary classification summary
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    f1_score,
)

# Final visual backbone builder
from ml.models.video.xception import build_xception_binary


@dataclass
class DatasetConfig:
    """
    Simple configuration object describing one dataset source.

    Each dataset contributes:
    - its name
    - extracted frame root folder
    - test split CSV file
    """
    name: str
    frames_root: Path
    test_csv: Path


# All datasets included in the final visual evaluation
DATASETS = [
    DatasetConfig(
        "FaceForensics++ C23",
        Path("data/interim/frames/FaceForensics++_C23"),
        Path("data/splits/faceforensics++_c23_test.csv"),
    ),
    DatasetConfig(
        "Celeb-DF v2",
        Path("data/interim/frames/Celeb-DF-v2"),
        Path("data/splits/celebdfv2_test.csv"),
    ),
    DatasetConfig(
        "DeeperForensics",
        Path("data/interim/frames/DeeperForensics"),
        Path("data/splits/deeperforensics_test.csv"),
    ),
    DatasetConfig(
        "FakeAVCeleb",
        Path("data/interim/frames/FakeAVCeleb_v1.2"),
        Path("data/splits/fakeavceleb_test.csv"),
    ),
]

# Output folder and checkpoint path for the final visual model
OUT_DIR = Path("experiments/results/final_visual_all_datasets_xception")
CKPT_PATH = OUT_DIR / "best_model.pt"

# Xception standard input resolution and evaluation settings
IMG_SIZE = 299
BATCH = 32
NUM_WORKERS = 0
MAX_FRAMES_PER_VIDEO = 5


def _torch_load_compat(path: Path, device: str):
    """
    Load a PyTorch checkpoint in a way that remains compatible
    across different torch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_transforms():
    """
    Build deterministic test-time preprocessing pipeline.

    Frames are resized to Xception input size and normalized
    with standard ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def read_split_labels(split_csv: Path) -> dict[str, int]:
    """
    Read mapping from original video_path -> label from a split CSV.

    This is later joined with SAFE_ID folders through _id_map.csv.
    """
    mapping = {}

    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vp = str(row.get("video_path", "")).replace("\\", "/").strip()
            if vp:
                mapping[vp] = int(row["label"])

    return mapping


def read_id_map(id_map_path: Path) -> dict[str, str]:
    """
    Read mapping from safe_id -> original video_path.

    This allows anonymized frame folders to be linked back
    to their correct labels.
    """
    mapping = {}

    with id_map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            safe_id = str(row["safe_id"]).strip()
            vpath = str(row["video_path"]).replace("\\", "/").strip()
            if safe_id and vpath:
                mapping[safe_id] = vpath

    return mapping


class MultiDatasetTestFrames(Dataset):
    """
    Combined test dataset built from all configured frame datasets.

    Each sample returns:
    - one frame tensor
    - binary label
    - unique video key
    - dataset name

    The video key combines dataset name + SAFE_ID so there is no clash
    between datasets that might reuse the same folder identifiers.
    """

    def __init__(self, transform, max_frames_per_video: int = 5):
        self.transform = transform
        self.samples = []

        # Iterate through all configured datasets and collect usable test frames
        for cfg in DATASETS:
            split_dir = cfg.frames_root / "test"
            id_map_path = split_dir / "_id_map.csv"

            # Skip datasets that are incomplete instead of failing immediately
            if not split_dir.exists() or not id_map_path.exists() or not cfg.test_csv.exists():
                print(f"[WARN] Skipping {cfg.name} due to missing test paths.")
                continue

            label_map = read_split_labels(cfg.test_csv)
            safe_to_vpath = read_id_map(id_map_path)

            for safe_id, vpath in safe_to_vpath.items():
                if vpath not in label_map:
                    continue

                label = int(label_map[vpath])
                video_dir = split_dir / safe_id

                if not video_dir.exists():
                    continue

                # Cap frames per video so very large folders do not dominate evaluation
                frame_paths = sorted(video_dir.glob("frame_*.jpg"))[:max_frames_per_video]

                if not frame_paths:
                    continue

                # Include dataset name in the key so each video remains globally unique
                video_key = f"{cfg.name}|{safe_id}"

                for frame_path in frame_paths:
                    self.samples.append((frame_path, label, video_key, cfg.name))

        if len(self.samples) == 0:
            raise FileNotFoundError("No test frames found.")

    def __len__(self):
        """Return total number of collected frame samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load one frame sample and return:
        - transformed image tensor
        - label
        - unique video key
        - dataset name
        """
        img_path, label, video_key, dataset_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, label, video_key, dataset_name


def compute_metrics(y_true, y_score, threshold=0.5):
    """
    Compute a full binary classification summary from labels and probabilities.

    This includes the main scalar metrics plus confusion matrix and
    classification report, making the output suitable for reporting
    and consolidation.
    """
    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    return {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "ap": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


@torch.no_grad()
def main():
    """
    Main evaluation pipeline for the final visual backbone.

    This function:
    - loads all available frame datasets
    - restores the trained Xception checkpoint
    - runs frame-level inference
    - aggregates frame predictions into video-level scores
    - computes both overall and per-dataset metrics
    - saves results as TXT and JSON
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Build combined evaluation dataset and loader
    ds = MultiDatasetTestFrames(make_transforms(), MAX_FRAMES_PER_VIDEO)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    # Rebuild model and load trained weights
    model = build_xception_binary(pretrained=False).to(device)
    ckpt = _torch_load_compat(CKPT_PATH, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Collect frame-level probabilities so they can later be averaged per video
    video_probs = defaultdict(list)
    video_labels = {}
    video_dataset = {}

    for x, y, video_keys, dataset_names in tqdm(loader, desc="Evaluating"):
        x = x.to(device)

        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

        for prob, label, video_key, dataset_name in zip(probs, y, video_keys, dataset_names):
            video_probs[video_key].append(float(prob))
            video_labels[video_key] = int(label)
            video_dataset[video_key] = dataset_name

    # Build overall and per-dataset video-level evaluation sets
    overall_true = []
    overall_score = []

    per_dataset = {}
    grouped = defaultdict(list)

    for video_key in sorted(video_probs.keys()):
        # Aggregate frame-level probabilities into one video-level score
        score = float(np.mean(video_probs[video_key]))
        label = video_labels[video_key]
        dataset_name = video_dataset[video_key]

        overall_true.append(label)
        overall_score.append(score)
        grouped[dataset_name].append((label, score))

    # Compute per-dataset metrics separately
    for dataset_name, rows in grouped.items():
        y_true = [r[0] for r in rows]
        y_score = [r[1] for r in rows]
        per_dataset[dataset_name] = compute_metrics(y_true, y_score, threshold=0.5)

    # Compute overall metrics across all datasets combined
    overall = compute_metrics(overall_true, overall_score, threshold=0.5)

    # Build structured JSON payload
    payload = {
        "dataset": "ALL frame datasets",
        "model": "Xception visual final",
        "checkpoint": str(CKPT_PATH),
        "aggregation": "mean(frame_probs_per_video)",
        "overall": overall,
        "per_dataset": per_dataset,
    }

    json_path = OUT_DIR / "test_metrics.json"
    txt_path = OUT_DIR / "test_report.txt"

    # Save JSON for machine-readable consolidation
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Save TXT report for human-readable inspection
    lines = [
        "FINAL VISUAL MODEL RESULTS (ALL DATASETS)\n",
        json.dumps(overall, indent=2),
        "\nPER DATASET:\n",
    ]

    for dataset_name, metrics in per_dataset.items():
        lines.append(f"{dataset_name}\n{json.dumps(metrics, indent=2)}\n")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("Overall accuracy:", overall["accuracy"])
    print("Overall balanced accuracy:", overall["balanced_accuracy"])
    print("Overall ROC-AUC:", overall["auc_roc"])
    print("Overall AP:", overall["ap"])
    print("Overall MCC:", overall["mcc"])
    print("Saved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()