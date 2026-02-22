"""
Evaluate ViT baseline on FaceForensics++ C23 frames (hashed folders).

Outputs:
- Accuracy, F1, AUC
- Balanced Accuracy, MCC
- Confusion matrix
- Classification report
- Saved report:
    experiments/results/ffpp_c23_vit_baseline/test_report.txt

Run:
    python -m ml.evaluate_vit
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef
)

from ml.models.video.vit import build_vit_binary


FRAMES_ROOT = Path("data/interim/frames/FaceForensics++_C23")
SPLITS_DIR = Path("data/splits")
TEST_CSV = SPLITS_DIR / "faceforensics++_c23_test.csv"

OUT_DIR = Path("experiments/results/ffpp_c23_vit_baseline")
CKPT_PATH = OUT_DIR / "best_model.pt"

IMG_SIZE = 224
BATCH = 32
NUM_WORKERS = 0


def _torch_load_compat(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def read_split_labels(split_csv: Path) -> dict[str, int]:
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
    def __init__(self, split: str, split_csv: Path, transform: transforms.Compose):
        self.transform = transform
        split_dir = FRAMES_ROOT / split
        id_map = split_dir / "_id_map.csv"
        if not id_map.exists():
            raise FileNotFoundError(f"Missing id map: {id_map}")

        vpath_to_label = read_split_labels(split_csv)
        safe_to_vpath = read_id_map(id_map)

        self.samples: list[tuple[Path, int]] = []
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
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y, str(img_path)


def main() -> None:
    out_report = OUT_DIR / "test_report.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = FFPPHashedFrameDataset("test", TEST_CSV, transform=make_transforms())
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    model = build_vit_binary(pretrained=False, img_size=IMG_SIZE).to(device)
    ckpt = _torch_load_compat(CKPT_PATH, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true: list[int] = []
    y_prob: list[float] = []

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(y.numpy().astype(int).tolist())
            y_prob.extend(probs.tolist())

    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    y_pred_np = (y_prob_np >= 0.5).astype(int)

    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np)
    try:
        auc = roc_auc_score(y_true_np, y_prob_np)
    except ValueError:
        auc = float("nan")

    bal_acc = balanced_accuracy_score(y_true_np, y_pred_np)
    mcc = matthews_corrcoef(y_true_np, y_pred_np)

    cm = confusion_matrix(y_true_np, y_pred_np)
    rep = classification_report(y_true_np, y_pred_np, digits=4)

    text = [
        "VIT RESULTS (FaceForensics++ C23 frame baseline)\n",
        f"Num samples: {len(y_true_np)}",
        f"Accuracy: {acc:.6f}",
        f"F1: {f1:.6f}",
        f"AUC: {auc:.6f}",
        f"Balanced Accuracy: {bal_acc:.6f}",
        f"MCC: {mcc:.6f}\n",
        "Confusion matrix:",
        str(cm),
        "\nClassification report:\n" + rep,
    ]

    out_report.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print("\nSaved report ->", out_report)


if __name__ == "__main__":
    main()
