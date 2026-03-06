"""
Evaluate FINAL visual backbone on ALL frame datasets.

Run:
    python -m ml.evaluate_final_visual_all_datasets

Outputs:
    experiments/results/final_visual_all_datasets_xception/test_report.txt
    experiments/results/final_visual_all_datasets_xception/test_metrics.json
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
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

from ml.models.video.xception import build_xception_binary


@dataclass
class DatasetConfig:
    name: str
    frames_root: Path
    test_csv: Path


DATASETS = [
    DatasetConfig("FaceForensics++ C23", Path("data/interim/frames/FaceForensics++_C23"), Path("data/splits/faceforensics++_c23_test.csv")),
    DatasetConfig("Celeb-DF v2", Path("data/interim/frames/Celeb-DF-v2"), Path("data/splits/celebdfv2_test.csv")),
    DatasetConfig("DeeperForensics", Path("data/interim/frames/DeeperForensics"), Path("data/splits/deeperforensics_test.csv")),
    DatasetConfig("FakeAVCeleb", Path("data/interim/frames/FakeAVCeleb_v1.2"), Path("data/splits/fakeavceleb_test.csv")),
]

OUT_DIR = Path("experiments/results/final_visual_all_datasets_xception")
CKPT_PATH = OUT_DIR / "best_model.pt"
IMG_SIZE = 299
BATCH = 32
NUM_WORKERS = 0
MAX_FRAMES_PER_VIDEO = 5


def _torch_load_compat(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def read_split_labels(split_csv: Path) -> dict[str, int]:
    mapping = {}
    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vp = str(row.get("video_path", "")).replace("\\", "/").strip()
            if vp:
                mapping[vp] = int(row["label"])
    return mapping


def read_id_map(id_map_path: Path) -> dict[str, str]:
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
    def __init__(self, transform, max_frames_per_video: int = 5):
        self.transform = transform
        self.samples = []

        for cfg in DATASETS:
            split_dir = cfg.frames_root / "test"
            id_map_path = split_dir / "_id_map.csv"
            if not split_dir.exists() or not id_map_path.exists() or not cfg.test_csv.exists():
                continue

            label_map = read_split_labels(cfg.test_csv)
            safe_to_vpath = read_id_map(id_map_path)

            for safe_id, vpath in safe_to_vpath.items():
                if vpath not in label_map:
                    continue
                label = int(label_map[vpath])
                video_dir = split_dir / safe_id
                frame_paths = sorted(video_dir.glob("frame_*.jpg"))[:max_frames_per_video]
                video_key = f"{cfg.name}|{safe_id}"
                for frame_path in frame_paths:
                    self.samples.append((frame_path, label, video_key, cfg.name))

        if len(self.samples) == 0:
            raise FileNotFoundError("No test frames found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, video_key, dataset_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, label, video_key, dataset_name


def compute_metrics(y_true, y_score, threshold=0.5):
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = MultiDatasetTestFrames(make_transforms(), MAX_FRAMES_PER_VIDEO)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    model = build_xception_binary(pretrained=False).to(device)
    ckpt = _torch_load_compat(CKPT_PATH, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

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

    overall_true = []
    overall_score = []

    per_dataset = {}
    grouped = defaultdict(list)
    for video_key in sorted(video_probs.keys()):
        score = float(np.mean(video_probs[video_key]))
        label = video_labels[video_key]
        dataset_name = video_dataset[video_key]
        overall_true.append(label)
        overall_score.append(score)
        grouped[dataset_name].append((label, score))

    for dataset_name, rows in grouped.items():
        y_true = [r[0] for r in rows]
        y_score = [r[1] for r in rows]
        per_dataset[dataset_name] = compute_metrics(y_true, y_score, threshold=0.5)

    overall = compute_metrics(overall_true, overall_score, threshold=0.5)

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
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["FINAL VISUAL MODEL RESULTS (ALL DATASETS)\n", json.dumps(overall, indent=2), "\nPER DATASET:\n"]
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
