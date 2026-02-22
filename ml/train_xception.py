"""
Train Visual CNN (Xception) on FaceForensics++ C23 sampled frames.

Supports hashed/safe-id frame folders created by:
pipelines/video/extract_frames_from_csv.py

Folder layout:
  data/interim/frames/FaceForensics++_C23/<split>/<SAFE_ID>/frame_000.jpg ...

Labels are obtained by joining:
  - split CSV (video_path -> label)
  - _id_map.csv in frames folder (safe_id -> video_path)

Run:
  python -m ml.train_xception
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from sklearn.metrics import roc_auc_score, f1_score

from ml.models.video.xception import build_xception_binary

from scripts.curve_writer import append_curve_row
from scripts.run_logger import log_run, log_exception


# ==============================
# CONFIG
# ==============================
DATA_ROOT = Path("data/interim/frames/FaceForensics++_C23")
OUT_DIR = Path("experiments/results/ffpp_c23_xception_baseline")

TRAIN_SPLIT_CSV = Path("data/splits/faceforensics++_c23_train.csv")
VAL_SPLIT_CSV = Path("data/splits/faceforensics++_c23_val.csv")

IMG_SIZE = 299
BATCH_TRAIN = 16
BATCH_VAL = 32
EPOCHS = 5
LR = 1e-4
NUM_WORKERS = 0  # safer on Windows/OneDrive; set 2 if stable
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _norm_path(s: str) -> str:
    # Normalize slashes for reliable matching
    return str(s).strip().replace("\\", "/").lower()


def _load_video_path_to_label(split_csv: Path) -> dict[str, int]:
    """
    Loads mapping: normalized video_path -> label (0/1) from your split CSV.
    Expects columns: video_path, label
    """
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    mapping: dict[str, int] = {}
    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Split CSV has no header: {split_csv}")

        if "video_path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"Split CSV must contain columns video_path,label. Found: {reader.fieldnames}"
            )

        for row in reader:
            vp = _norm_path(row["video_path"])
            lab = int(row["label"])
            mapping[vp] = lab

    if not mapping:
        raise ValueError(f"No rows loaded from split CSV: {split_csv}")

    return mapping


def _load_safeid_to_label_from_idmap(frames_split_dir: Path, split_csv: Path) -> dict[str, int]:
    """
    Builds mapping: safe_id -> label by joining:
      frames_split_dir/_id_map.csv  (safe_id, video_id, video_path)
      split_csv                    (video_path, label, ...)
    """
    id_map_path = frames_split_dir / "_id_map.csv"
    if not id_map_path.exists():
        raise FileNotFoundError(
            f"Missing {id_map_path}. Re-run extract_frames_from_csv.py for this split to generate it."
        )

    path_to_label = _load_video_path_to_label(split_csv)

    safeid_to_label: dict[str, int] = {}
    missing = 0
    example = None

    with id_map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"_id_map.csv has no header: {id_map_path}")

        needed = {"safe_id", "video_path"}
        if not needed.issubset(set(reader.fieldnames)):
            raise ValueError(f"_id_map.csv must contain columns: safe_id,video_path. Found: {reader.fieldnames}")

        for row in reader:
            sid = str(row["safe_id"]).strip()
            vp = _norm_path(row["video_path"])

            if vp in path_to_label:
                safeid_to_label[sid] = int(path_to_label[vp])
            else:
                missing += 1
                if example is None:
                    example = row

    if not safeid_to_label:
        raise ValueError(
            f"Join produced 0 labels for split {frames_split_dir}. "
            f"Check that video_path strings in split CSV match _id_map.csv.\n"
            f"Example missing row: {example}"
        )

    if missing > 0:
        print(f"[WARN] {missing} id_map rows had video_path not found in split CSV (ok if id_map contains extra).")

    return safeid_to_label


class FFPPFramesSafeIdDataset(Dataset):
    """
    Uses SAFE_ID folder names to look up labels from safeid_to_label.
    Structure:
      root_dir/<SAFE_ID>/frame_000.jpg ...
    """

    def __init__(self, root_dir: Path, safeid_to_label: dict[str, int], transform=None, strict: bool = True):
        self.root_dir = Path(root_dir)
        self.safeid_to_label = safeid_to_label
        self.transform = transform
        self.strict = strict

        image_paths = []
        image_paths.extend(self.root_dir.rglob("*.jpg"))
        image_paths.extend(self.root_dir.rglob("*.jpeg"))
        image_paths.extend(self.root_dir.rglob("*.png"))
        image_paths = sorted(image_paths)

        if not image_paths:
            raise FileNotFoundError(f"No frames found under: {self.root_dir}")

        filtered = []
        missing = 0
        example_missing = None

        for p in image_paths:
            safe_id = p.parent.name
            if safe_id in self.safeid_to_label:
                filtered.append(p)
            else:
                missing += 1
                if example_missing is None:
                    example_missing = p

        if strict and missing > 0:
            raise KeyError(
                f"{missing} frame(s) have SAFE_ID not found in labels map. "
                f"Example path: {example_missing}. "
                "This means your frames were extracted with an _id_map.csv that doesn't match the split CSV."
            )

        self.image_paths = filtered
        if not self.image_paths:
            raise ValueError(
                f"After filtering by labels map, dataset is empty for root: {self.root_dir}."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        safe_id = path.parent.name
        label = float(self.safeid_to_label[safe_id])

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label), safe_id


def main():
    run_name = "xception_baseline"
    curve_csv = f"experiments/logs/training_curves/{run_name}.csv"

    try:
        set_seed(SEED)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        best_path = OUT_DIR / "best_model.pt"

        train_dir = DATA_ROOT / "train"
        val_dir = DATA_ROOT / "val"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", device)

        # Build safe_id -> label using _id_map.csv join on video_path
        train_labels = _load_safeid_to_label_from_idmap(train_dir, TRAIN_SPLIT_CSV)
        val_labels = _load_safeid_to_label_from_idmap(val_dir, VAL_SPLIT_CSV)

        train_ds = FFPPFramesSafeIdDataset(
            train_dir, safeid_to_label=train_labels, transform=make_transforms(train=True), strict=True
        )
        val_ds = FFPPFramesSafeIdDataset(
            val_dir, safeid_to_label=val_labels, transform=make_transforms(train=False), strict=True
        )

        print("Train frames:", len(train_ds))
        print("Val frames:", len(val_ds))

        train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False, num_workers=NUM_WORKERS)

        model = build_xception_binary(pretrained=True).to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        best_val_loss = float("inf")
        best_epoch = 0
        best_val_auc = float("nan")
        best_val_f1 = float("nan")

        for epoch in range(1, EPOCHS + 1):
            # ---- train ----
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for images, labels, safe_ids in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - train"):
                images = images.to(device)
                labels = labels.float().to(device)

                logits = model(images).squeeze(1)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = images.size(0)
                train_loss_sum += loss.item() * bs
                train_count += bs

            train_loss = train_loss_sum / max(train_count, 1)

            # ---- val ----
            model.eval()
            val_loss_sum = 0.0
            val_count = 0

            all_labels = []
            all_probs = []

            with torch.no_grad():
                for images, labels, safe_ids in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - val"):
                    images = images.to(device)
                    labels = labels.float().to(device)

                    logits = model(images).squeeze(1)
                    loss = loss_fn(logits, labels)

                    probs = torch.sigmoid(logits)

                    bs = images.size(0)
                    val_loss_sum += loss.item() * bs
                    val_count += bs

                    all_labels.extend(labels.detach().cpu().numpy().tolist())
                    all_probs.extend(probs.detach().cpu().numpy().tolist())

            val_loss = val_loss_sum / max(val_count, 1)

            val_auc = float("nan")
            if len(set(all_labels)) > 1:
                val_auc = roc_auc_score(all_labels, all_probs)

            preds = (np.array(all_probs) >= 0.5).astype(int).tolist()
            val_f1 = f1_score(all_labels, preds, zero_division=0)

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_auc={val_auc:.4f}  val_f1={val_f1:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_val_auc = val_auc
                best_val_f1 = val_f1

                torch.save({
                    "model_state": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_auc": best_val_auc,
                    "val_f1": best_val_f1,
                    "epoch": best_epoch,
                    "config": {
                        "model": "xception",
                        "img_size": IMG_SIZE,
                        "epochs": EPOCHS,
                        "lr": LR,
                        "seed": SEED,
                        "batch_train": BATCH_TRAIN,
                        "batch_val": BATCH_VAL,
                        "num_workers": NUM_WORKERS,
                        "data_root": str(DATA_ROOT),
                        "train_split_csv": str(TRAIN_SPLIT_CSV),
                        "val_split_csv": str(VAL_SPLIT_CSV),
                        "label_source": "_id_map.csv join on video_path",
                    }
                }, best_path)
                print("Saved best ->", best_path)

            append_curve_row(curve_csv, {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_f1": val_f1,
            })

        log_run(run_name, payload={
            "task": "train",
            "model": "xception",
            "dataset": "FaceForensics++_C23",
            "split": "train/val",
            "seed": SEED,
            "metrics": {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_val_auc": best_val_auc,
                "best_val_f1": best_val_f1,
            },
            "artifacts": [str(best_path), str(curve_csv)],
            "notes": "Frame folders use SAFE_ID from extractor. Labels joined via _id_map.csv + split CSV video_path.",
        })

        print("Done. Best val loss:", best_val_loss)

    except Exception as exc:
        log_exception(run_name, exc, context={"script": "ml/train_xception.py"})
        raise


if __name__ == "__main__":
    main()
