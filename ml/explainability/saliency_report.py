"""
Saliency Report Generator (Grad-CAM wrapper)
============================================

This script automates the generation of a small explainability report
using Grad-CAM outputs from the trained Xception deepfake classifier.
Its purpose is to produce visual examples and a short summary pack
that can be used for dissertation discussion and examiner review.

The script:
- loads labelled frame folders for a chosen data split
- selects samples using different strategies
- runs Grad-CAM on one frame per selected sample
- saves overlay images and grayscale heatmaps
- generates both CSV and Markdown reports summarising the results

This utility complements:
    ml/explainability/gradcam.py

Supports:
- Xception binary classifier (frame-based)

Folder assumptions
------------------
Frames extracted using pipelines/video/extract_frames_from_csv.py:

data/interim/frames/FaceForensics++_C23/<split>/<SAFE_ID>/frame_000.jpg ...

Labels:
- from split CSV (video_path,label,video_id,...) joined via <split>/_id_map.csv
  (same approach used in the training pipeline)

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

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# argparse is used to make the script configurable from the command line
import argparse

# csv is used to read split files and write the final report table
import csv

# random is used for reproducible sample selection
import random

# dataclass provides a clean structure for storing per-sample results
from dataclasses import dataclass

# Path is used for safer and clearer file/folder handling
from pathlib import Path

# Optional is used where a value may legitimately be missing
from typing import Optional

# tqdm provides progress bars for longer loops
from tqdm import tqdm

# NumPy is used for array handling and heatmap conversion
import numpy as np

# PyTorch is used for model loading and inference
import torch

# PIL is used for reading and saving images
from PIL import Image

# torchvision transforms are used to prepare input images for the Xception model
from torchvision import transforms

# Import the Grad-CAM utility and helper for locating the last convolutional layer
from ml.explainability.gradcam import GradCAM, find_last_conv_layer

# Import the binary Xception architecture used in this project
from ml.models.video.xception import build_xception_binary


# -----------------------------
# Config defaults
# -----------------------------

# Default location of extracted frame folders
DEFAULT_DATA_ROOT = Path("data/interim/frames/FaceForensics++_C23")

# Default location of dataset split CSV files
DEFAULT_SPLITS_DIR = Path("data/splits")

# Default checkpoint of the trained Xception baseline model
DEFAULT_CKPT = Path("experiments/results/ffpp_c23_xception_baseline/best_model.pt")

# Default root output directory for explainability reports
DEFAULT_OUT_ROOT = Path("experiments/explainability/xception_baseline")

# Input size required by the Xception classifier
IMG_SIZE = 299

# Fixed random seed to make selection reproducible
SEED = 42


# -----------------------------
# Helper functions
# -----------------------------
def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.
    This helps ensure sample selection is consistent across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform() -> transforms.Compose:
    """
    Build the image preprocessing pipeline used before model inference.

    The transform mirrors ImageNet-style preprocessing expected by
    the Xception backbone: resize, convert to tensor, then normalize.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _norm_path(s: str) -> str:
    """
    Normalize file paths to improve matching consistency across
    operating systems and CSV formatting differences.
    """
    return str(s).strip().replace("\\", "/").lower()


def load_video_path_to_label(split_csv: Path) -> dict[str, int]:
    """
    Load a mapping from normalized video path to ground-truth label.

    This is used as the first part of the join process when linking
    SAFE_ID frame folders back to their labels.
    """
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    mapping: dict[str, int] = {}
    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate that the CSV has the required structure
        if reader.fieldnames is None:
            raise ValueError(f"Split CSV has no header: {split_csv}")
        if "video_path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"Split CSV must contain video_path,label. Found: {reader.fieldnames}")

        # Build the lookup dictionary
        for row in reader:
            vp = _norm_path(row["video_path"])
            mapping[vp] = int(row["label"])

    if not mapping:
        raise ValueError(f"No rows read from: {split_csv}")

    return mapping


def load_safeid_to_label(frames_split_dir: Path, split_csv: Path) -> dict[str, int]:
    """
    Build a mapping from SAFE_ID folder name to label.

    This is done by joining:
    - _id_map.csv   -> safe_id, video_path
    - split CSV     -> video_path, label

    This approach keeps frame folders anonymous while still preserving
    access to the correct ground-truth labels.
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

        # Validate the mapping file structure
        if reader.fieldnames is None:
            raise ValueError(f"_id_map.csv has no header: {id_map_path}")
        if "safe_id" not in reader.fieldnames or "video_path" not in reader.fieldnames:
            raise ValueError(f"_id_map.csv must contain safe_id,video_path. Found: {reader.fieldnames}")

        # Join safe_id to label through the normalized video path
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
    """
    Return all SAFE_ID folder names inside the selected split directory.
    Each folder corresponds to one video/sample.
    """
    return sorted([p.name for p in frames_split_dir.iterdir() if p.is_dir()])


def choose_frame_for_safe_id(safe_dir: Path, preferred: str = "frame_000.jpg") -> Optional[Path]:
    """
    Choose one representative frame for a given SAFE_ID folder.

    By default, the script prefers frame_000.jpg for consistency.
    If that frame is missing, it falls back to the first available frame.
    """
    p = safe_dir / preferred
    if p.exists():
        return p

    frames = sorted(safe_dir.glob("frame_*.jpg"))
    return frames[0] if frames else None


def load_rgb(path: Path) -> np.ndarray:
    """
    Load an image as an RGB NumPy array.

    PIL already returns RGB when explicitly converted, which avoids
    channel-order issues that can happen with OpenCV.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img)


def to_input_tensor(rgb: np.ndarray, tfm: transforms.Compose, device: str) -> torch.Tensor:
    """
    Convert an RGB NumPy image into a model-ready input tensor.

    The returned tensor has shape [1, 3, H, W], including the batch dimension.
    """
    pil = Image.fromarray(rgb)
    x = tfm(pil).unsqueeze(0).to(device)
    return x


def sigmoid_prob(logit: torch.Tensor) -> float:
    """
    Convert a raw binary logit into a probability in the range [0,1].
    """
    return float(torch.sigmoid(logit).detach().cpu().item())


@dataclass
class SampleResult:
    """
    Container for all information recorded for one explained sample.

    This keeps reporting cleaner and makes CSV/Markdown generation easier.
    """
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
    Select which samples to explain.

    Available strategies:
    - random:       random subset
    - top_fp:       most confident false positives
    - top_fn:       most confident false negatives
    - top_correct:  most confident correct predictions

    This makes it possible to generate explainability outputs for
    different evaluation perspectives, not only random examples.
    """
    eligible = [sid for sid in safe_ids if sid in safe_to_label]

    if not eligible:
        raise ValueError("No eligible samples found (safe_ids ∩ labels map is empty).")

    if scored is None or strategy == "random":
        random.shuffle(eligible)
        return eligible[:n]

    # Select the most confident false positives:
    # real samples predicted as fake with high fake probability
    if strategy == "top_fp":
        fps = [r for r in scored if r.label == 0 and r.pred == 1]
        fps.sort(key=lambda r: r.prob_fake, reverse=True)
        return [r.safe_id for r in fps[:n]]

    # Select the most confident false negatives:
    # fake samples predicted as real with very low fake probability
    if strategy == "top_fn":
        fns = [r for r in scored if r.label == 1 and r.pred == 0]
        fns.sort(key=lambda r: r.prob_fake)
        return [r.safe_id for r in fns[:n]]

    # Select the most confident correct predictions
    if strategy == "top_correct":
        correct = [r for r in scored if r.is_correct == 1]
        correct.sort(key=lambda r: abs(r.prob_fake - 0.5), reverse=True)
        return [r.safe_id for r in correct[:n]]

    raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    """
    Main pipeline for saliency report generation.

    It:
    - reads command-line parameters
    - validates inputs
    - loads the trained model
    - selects samples
    - runs Grad-CAM
    - saves visual outputs and summary reports
    """
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

    # Set random seeds before sampling and inference
    set_seed(SEED)

    # Resolve paths and create output folders
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

    # Build expected split CSV filename
    split_csv = splits_dir / f"faceforensics++_c23_{split}.csv"

    # Validate required inputs before doing any expensive work
    if not split_csv.exists():
        raise FileNotFoundError(f"Expected split CSV: {split_csv}")

    if not frames_split_dir.exists():
        raise FileNotFoundError(f"Frames split folder not found: {frames_split_dir}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Select device automatically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Split:", split)
    print("Frames:", frames_split_dir)
    print("Split CSV:", split_csv)
    print("Checkpoint:", ckpt_path)

    # Build SAFE_ID -> label lookup
    safe_to_label = load_safeid_to_label(frames_split_dir, split_csv)

    # Load trained Xception model
    model = build_xception_binary(pretrained=False).to(device)

    # Load checkpoint safely while remaining compatible with older torch versions
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    # Some checkpoints store model_state inside a dictionary, others may be raw state dicts
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    # Prepare Grad-CAM on the last convolutional layer
    target_layer = find_last_conv_layer(model)
    cam = GradCAM(model, target_layer=target_layer)

    # Build the image preprocessing pipeline
    tfm = make_transform()

    # List all available SAFE_ID folders
    safe_ids = list_safe_ids(frames_split_dir)

    # Optional quick scoring pass:
    # needed for strategies that depend on confidence
    scored_results: Optional[list[SampleResult]] = None
    if args.strategy != "random":
        print(f"[INFO] Running quick scoring pass for strategy={args.strategy} ...")
        scored_results = []

        # Limit the scoring pool for speed, especially on CPU
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

                # Standardize output shape before converting to probability
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

    # Select final set of samples to explain
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

        # First compute normal model prediction for reporting
        with torch.no_grad():
            logit = model(x)
            if logit.ndim == 2 and logit.shape[1] == 1:
                logit = logit[0, 0]
            else:
                logit = logit.squeeze()[0] if logit.ndim > 0 else logit
            prob = sigmoid_prob(logit)

        pred = 1 if prob >= args.threshold else 0
        label = int(safe_to_label[sid])

        # Grad-CAM requires gradients, so this call runs without no_grad
        result = cam(input_tensor=x, original_rgb=rgb, target_class=None)

        # Include prediction metadata in filenames to make saved outputs easier to inspect
        overlay_path = overlays_dir / f"{sid}_p{prob:.3f}_y{label}.png"
        heatmap_path = heatmaps_dir / f"{sid}_p{prob:.3f}_y{label}.png"

        # Save blended overlay image
        Image.fromarray(result.overlay_rgb).save(overlay_path)

        # Save grayscale heatmap separately for analysis
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

    # Remove Grad-CAM hooks after use
    cam.close()

    # Write CSV report for structured downstream analysis
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
                r.safe_id,
                r.frame_path,
                r.label,
                f"{r.prob_fake:.6f}",
                r.pred,
                r.is_correct,
                r.overlay_path,
                r.heatmap_path,
            ])

    # Write a short Markdown report for human-readable summary
    report_md = out_root / "report.md"
    n_total = len(results)
    n_correct = sum(r.is_correct for r in results)
    acc = (n_correct / n_total) if n_total else 0.0
    n_real = sum(1 for r in results if r.label == 0)
    n_fake = sum(1 for r in results if r.label == 1)

    lines = [
        f"# VerifAI Saliency Report (Grad-CAM)\n",
        f"- Split: `{split}`\n",
        f"- Strategy: `{args.strategy}`\n",
        f"- Samples explained: **{n_total}**\n",
        f"- Class balance in explained set: real={n_real}, fake={n_fake}\n",
        f"- Accuracy on explained set (threshold={args.threshold}): **{acc:.3f}**\n",
        "\n## Saved Overlays\n"
    ]

    for r in results:
        rel = Path(r.overlay_path).relative_to(out_root)
        lines.append(f"- `{r.safe_id}` | y={r.label} pred={r.pred} p_fake={r.prob_fake:.3f} | {rel}\n")

    report_md.write_text("".join(lines), encoding="utf-8")

    # Print output locations for quick confirmation
    print("[OK] Wrote:", report_csv)
    print("[OK] Wrote:", report_md)
    print("[OK] Overlays:", overlays_dir)
    print("[OK] Heatmaps:", heatmaps_dir)


if __name__ == "__main__":
    main()