"""
Evaluate Visual CNN (Xception) on FaceForensics++ C23 (VIDEO-LEVEL metrics).

What it does
------------
- Loads FF++ C23 test frames (hashed folders) via FFPPFrameDataset
- Loads checkpoint: experiments/results/ffpp_c23_xception_baseline/best_model.pt
- Runs frame inference -> aggregates to video score by mean(frame_prob)
- Computes and saves consistent metrics for consolidation:
    - Accuracy
    - Balanced Accuracy
    - F1 (weighted + macro via sklearn report inside compute_binary_metrics)
    - ROC-AUC
    - Average Precision (AP)
    - MCC
    - Confusion matrix + classification report
- Saves:
    experiments/results/ffpp_c23_xception_baseline/test_report.txt
    experiments/results/ffpp_c23_xception_baseline/test_metrics.json

Run:
    python -m ml.evaluate_xception
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ml.data_loader import FFPPFrameDataset
from ml.models.video.xception import build_xception_binary
from ml.metrics import (
    aggregate_video_scores_mean,
    compute_binary_metrics,
    save_metrics_report,
)


def make_transforms():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _torch_load_compat(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def main():
    data_root = Path("data/interim/frames/FaceForensics++_C23")
    test_dir = data_root / "test"

    ckpt_path = Path("experiments/results/ffpp_c23_xception_baseline/best_model.pt")
    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # IMPORTANT: your FFPPFrameDataset needs mapping_file for hashed-folder layout
    # (same logic you used in your other scripts)
    ds = FFPPFrameDataset(
        str(test_dir),
        transform=make_transforms(),
        mapping_file=r"data\interim\frames\FaceForensics++_C23\test\_id_map.csv",
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_xception_binary(pretrained=False).to(device)
    ckpt = _torch_load_compat(ckpt_path, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # frame -> prob, then aggregate by mean(prob) per video_id
    video_probs = defaultdict(list)
    video_label = {}

    for images, labels, video_ids in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        for p, y, vid in zip(probs, labels, video_ids):
            vid = str(vid)
            y = int(y)
            video_probs[vid].append(float(p))
            if vid not in video_label:
                video_label[vid] = y

    # Use your shared helper for consistent aggregation output ordering
    y_true, y_score, vids = aggregate_video_scores_mean(video_probs, video_label)

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        include_curves=True,  # ensures ROC-AUC + AP are computed when possible
    )

    print("\nVIDEO-LEVEL RESULTS (FF++ C23 Xception baseline)")
    print("Num videos:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)

    txt_path, json_path = save_metrics_report(
        out_dir=out_dir,
        name="test",
        metrics=metrics,
        extra={
            "dataset": "FaceForensics++ C23",
            "aggregation": "mean(frame_probs)",
            "checkpoint": str(ckpt_path),
            "split": "test",
            "model": "Xception baseline",
        },
    )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()