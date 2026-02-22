"""
Saliency Report Generator (Grad-CAM wrapper)
============================================

Purpose
-------
Runs Grad-CAM explainability over a set of samples and writes a small
"report pack" suitable for dissertation/markers:

- overlay images (original + heatmap)
- report.csv with per-sample predictions, labels, confidence
- report.md summarizing results and listing saved examples

This script complements:
    ml/explainability/gradcam.py

Supports (v1):
- Xception binary classifier (frame-based)

Folder assumptions
------------------
Frames extracted using pipelines/video/extract_frames_from_csv.py:

data/interim/frames/FaceForensics++_C23/<split>/<SAFE_ID>/frame_000.jpg ...

Labels:
- from split CSV (video_path,label,video_id,...) joined via <split>/_id_map.csv
  (same approach used in your updated train_xception.py).

Run
---
From project root:
    python -m ml.explainability.saliency_report --split val --n 20

Optional:
    python -m ml.explainability.saliency_report --split val --n 40 --strategy top_fp

Outputs
-------
experiments/explainability/xception_baseline/<split>/
  overlays/
  heatmaps/
  report.csv
  report.md

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.explainability.gradcam import GradCAM, find_last_conv_layer
from ml.models.video.xception import build_xception_binary

# -----------------------------
# Config defaults (safe)
# -----------------------------
DEFAULT_DATA_ROOT = Path("data/interim/frames/FaceForensics++_C23")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_CKPT = Path("experiments/results/ffpp_c23_xception_baseline/best_model.pt")
DEFAULT_OUT_ROOT = Path("experiments/explainability/xception_baseline")

IMG_SIZE = 299
SEED = 42


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _norm_path(s: str) -> str:
    return str(s).strip().replace("\\", "/").lower()


def load_video_path_to_label(split_csv: Path) -> dict[str, int]:
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    mapping: dict[str, int] = {}
    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Split CSV has no header: {split_csv}")
        if "video_path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"Split CSV must contain video_path,label. Found: {reader.fieldnames}")

        for row in reader:
            vp = _norm_path(row["video_path"])
            mapping[vp] = int(row["label"])

    if not mapping:
        raise ValueError(f"No rows read from: {split_csv}")
    return mapping


def load_safeid_to_label(frames_split_dir: Path, split_csv: Path) -> dict[str, int]:
    """
    Join frames_split_dir/_id_map.csv (safe_id, video_path) with split_csv (video_path,label).
    """
    id_map_path = frames_split_dir / "_id_map.csv"
    if not id_map_path.exists():
        raise FileNotFoundError(
            f"Missing {id_map_path}. Re-run extract_frames_from_csv.py to generate it."
        )

    path_to_label = load_video_path_to_label(split_csv)

    safe_to_label: dict[str, int] = {}
    with id_map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"_id_map.csv has no header: {id_map_path}")
        if "safe_id" not in reader.fieldnames or "video_path" not in reader.fieldnames:
            raise ValueError(f"_id_map.csv must contain safe_id,video_path. Found: {reader.fieldnames}")

        for row in reader:
            sid = str(row["safe_id"]).strip()
            vp = _norm_path(row["video_path"])
            if vp in path_to_label:
                safe_to_label[sid] = int(path_to_label[vp])

    if not safe_to_label:
        raise ValueError(
            f"No safe_id labels could be built for {frames_split_dir}. "
            "Likely mismatch between _id_map.csv video_path and split CSV video_path."
        )

    return safe_to_label


def list_safe_ids(frames_split_dir: Path) -> list[str]:
    return sorted([p.name for p in frames_split_dir.iterdir() if p.is_dir()])


def choose_frame_for_safe_id(safe_dir: Path, preferred: str = "frame_000.jpg") -> Optional[Path]:
    p = safe_dir / preferred
    if p.exists():
        return p
    frames = sorted(safe_dir.glob("frame_*.jpg"))
    return frames[0] if frames else None


def load_rgb(path: Path) -> np.ndarray:
    # PIL gives RGB already
    img = Image.open(path).convert("RGB")
    return np.array(img)


def to_input_tensor(rgb: np.ndarray, tfm: transforms.Compose, device: str) -> torch.Tensor:
    pil = Image.fromarray(rgb)
    x = tfm(pil).unsqueeze(0).to(device)  # [1,3,H,W]
    return x


def sigmoid_prob(logit: torch.Tensor) -> float:
    return float(torch.sigmoid(logit).detach().cpu().item())


@dataclass
class SampleResult:
    safe_id: str
    frame_path: str
    label: int
    prob_fake: float
    pred: int
    is_correct: int
    overlay_path: str
    heatmap_path: str


def pick_samples(
    safe_ids: list[str],
    safe_to_label: dict[str, int],
    scored: Optional[list[SampleResult]],
    n: int,
    strategy: str,
) -> list[str]:
    """
    If scored is None: just random sample of safe_ids that exist in safe_to_label.
    If scored exists: pick based on strategy using scored results (after quick scoring pass).
    """
    eligible = [sid for sid in safe_ids if sid in safe_to_label]

    if not eligible:
        raise ValueError("No eligible samples found (safe_ids ∩ labels map is empty).")

    if scored is None or strategy == "random":
        random.shuffle(eligible)
        return eligible[:n]

    # scored-based strategies (use already computed probs/preds)
    if strategy == "top_fp":
        # label=0 but predicted fake with high prob
        fps = [r for r in scored if r.label == 0 and r.pred == 1]
        fps.sort(key=lambda r: r.prob_fake, reverse=True)
        return [r.safe_id for r in fps[:n]]

    if strategy == "top_fn":
        # label=1 but predicted real (low prob_fake)
        fns = [r for r in scored if r.label == 1 and r.pred == 0]
        fns.sort(key=lambda r: r.prob_fake)  # lowest prob_fake first
        return [r.safe_id for r in fns[:n]]

    if strategy == "top_correct":
        correct = [r for r in scored if r.is_correct == 1]
        correct.sort(key=lambda r: abs(r.prob_fake - 0.5), reverse=True)  # most confident correct
        return [r.safe_id for r in correct[:n]]

    raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--n", type=int, default=20, help="Number of videos to explain (one frame each)")
    parser.add_argument("--strategy", choices=["random", "top_fp", "top_fn", "top_correct"], default="random")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for fake prediction")
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--splits_dir", type=str, default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    set_seed(SEED)

    split = args.split
    frames_split_dir = Path(args.data_root) / split
    splits_dir = Path(args.splits_dir)
    ckpt_path = Path(args.ckpt)
    out_root = Path(args.out_root) / split
    overlays_dir = out_root / "overlays"
    heatmaps_dir = out_root / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    split_csv = splits_dir / f"faceforensics++_c23_{split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Expected split CSV: {split_csv}")

    if not frames_split_dir.exists():
        raise FileNotFoundError(f"Frames split folder not found: {frames_split_dir}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Split:", split)
    print("Frames:", frames_split_dir)
    print("Split CSV:", split_csv)
    print("Checkpoint:", ckpt_path)

    # Build labels
    safe_to_label = load_safeid_to_label(frames_split_dir, split_csv)

    # Load model
    model = build_xception_binary(pretrained=False).to(device)
    # PyTorch 2.6+ defaults weights_only=True (safer), but our checkpoint is a trusted dict
    # created by this project (may include numpy scalars). Load with weights_only=False.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # older torch versions don't have weights_only
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    # Grad-CAM setup
    target_layer = find_last_conv_layer(model)
    cam = GradCAM(model, target_layer=target_layer)

    tfm = make_transform()

    safe_ids = list_safe_ids(frames_split_dir)

    # Optional: quick scoring pass if using non-random strategy
    scored_results: Optional[list[SampleResult]] = None
    if args.strategy != "random":
        print(f"[INFO] Running quick scoring pass for strategy={args.strategy} ...")
        scored_results = []
        # score on at most 500 to keep it fast on CPU
        eligible = [sid for sid in safe_ids if sid in safe_to_label]
        random.shuffle(eligible)
        eligible = eligible[: min(len(eligible), 500)]

        with torch.no_grad():
            for sid in tqdm(eligible, desc="scoring"):
                safe_dir = frames_split_dir / sid
                frame_path = choose_frame_for_safe_id(safe_dir)
                if frame_path is None:
                    continue
                rgb = load_rgb(frame_path)
                x = to_input_tensor(rgb, tfm, device)
                logit = model(x)
                if logit.ndim == 2 and logit.shape[1] == 1:
                    logit = logit[0, 0]
                else:
                    logit = logit.squeeze()[0] if logit.ndim > 0 else logit
                prob = sigmoid_prob(logit)
                pred = 1 if prob >= args.threshold else 0
                label = int(safe_to_label[sid])
                scored_results.append(
                    SampleResult(
                        safe_id=sid,
                        frame_path=str(frame_path),
                        label=label,
                        prob_fake=prob,
                        pred=pred,
                        is_correct=int(pred == label),
                        overlay_path="",
                        heatmap_path="",
                    )
                )

    chosen = pick_samples(safe_ids, safe_to_label, scored_results, args.n, args.strategy)
    if not chosen:
        raise ValueError("No samples selected. Try a different strategy or increase scoring pool.")

    print(f"[OK] Selected {len(chosen)} samples using strategy={args.strategy}")

    results: list[SampleResult] = []

    # Main explainability loop
    for sid in tqdm(chosen, desc="gradcam"):
        safe_dir = frames_split_dir / sid
        frame_path = choose_frame_for_safe_id(safe_dir)
        if frame_path is None:
            continue

        rgb = load_rgb(frame_path)
        x = to_input_tensor(rgb, tfm, device)

        # forward for prob/pred
        with torch.no_grad():
            logit = model(x)
            if logit.ndim == 2 and logit.shape[1] == 1:
                logit = logit[0, 0]
            else:
                logit = logit.squeeze()[0] if logit.ndim > 0 else logit
            prob = sigmoid_prob(logit)

        pred = 1 if prob >= args.threshold else 0
        label = int(safe_to_label[sid])

        # gradcam needs gradients -> enable grad for this call
        # (model is in eval mode but gradients are allowed)
        result = cam(input_tensor=x, original_rgb=rgb, target_class=None)

        overlay_path = overlays_dir / f"{sid}_p{prob:.3f}_y{label}.png"
        heatmap_path = heatmaps_dir / f"{sid}_p{prob:.3f}_y{label}.png"

        Image.fromarray(result.overlay_rgb).save(overlay_path)
        # Save heatmap as grayscale 0-255
        hm = np.uint8(255 * result.heatmap)
        Image.fromarray(hm).save(heatmap_path)

        results.append(
            SampleResult(
                safe_id=sid,
                frame_path=str(frame_path),
                label=label,
                prob_fake=float(prob),
                pred=int(pred),
                is_correct=int(pred == label),
                overlay_path=str(overlay_path),
                heatmap_path=str(heatmap_path),
            )
        )

    cam.close()

    # Write CSV report
    report_csv = out_root / "report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "safe_id",
            "frame_path",
            "label",
            "prob_fake",
            "pred",
            "is_correct",
            "overlay_path",
            "heatmap_path",
        ])
        for r in results:
            writer.writerow([
                r.safe_id, r.frame_path, r.label, f"{r.prob_fake:.6f}", r.pred, r.is_correct, r.overlay_path, r.heatmap_path
            ])

    # Write Markdown summary
    report_md = out_root / "report.md"
    n_total = len(results)
    n_correct = sum(r.is_correct for r in results)
    acc = (n_correct / n_total) if n_total else 0.0
    n_real = sum(1 for r in results if r.label == 0)
    n_fake = sum(1 for r in results if r.label == 1)

    lines = [f"# VerifAI Saliency Report (Grad-CAM)\n", f"- Split: `{split}`\n", f"- Strategy: `{args.strategy}`\n",
             f"- Samples explained: **{n_total}**\n",
             f"- Class balance in explained set: real={n_real}, fake={n_fake}\n",
             f"- Accuracy on explained set (threshold={args.threshold}): **{acc:.3f}**\n", "\n## Saved Overlays\n"]
    for r in results:
        rel = Path(r.overlay_path).relative_to(out_root)
        lines.append(f"- `{r.safe_id}` | y={r.label} pred={r.pred} p_fake={r.prob_fake:.3f} | {rel}\n")

    report_md.write_text("".join(lines), encoding="utf-8")

    print("[OK] Wrote:", report_csv)
    print("[OK] Wrote:", report_md)
    print("[OK] Overlays:", overlays_dir)
    print("[OK] Heatmaps:", heatmaps_dir)


if __name__ == "__main__":
    main()
