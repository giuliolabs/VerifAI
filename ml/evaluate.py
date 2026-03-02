"""
Evaluate MobileNetV2 baseline on FaceForensics++ C23 (video-level)

Run (from project root):
    python ml/evaluate.py

Outputs (in experiments/results/ffpp_c23_mobilenet_baseline):
    - video_level_report.txt
    - video_level_metrics.json
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from ml.augmentations import get_transforms
from ml.data_loader import FFPPFrameDataset
from ml.models.video.mobilenet_baseline import build_model
from ml.metrics import (
    aggregate_video_scores_mean,
    compute_binary_metrics,
    save_metrics_report,
)


@torch.no_grad()
def main():
    data_root = Path("data/interim/frames/FaceForensics++_C23")
    out_dir = Path("experiments/results/ffpp_c23_mobilenet_baseline")
    model_path = out_dir / "best_model.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run ml/train.py first.")

    test_dir = data_root / "test"
    test_ds = FFPPFrameDataset(
        str(test_dir),
        transform=get_transforms(train=False),
        mapping_file=r"data\interim\frames\FaceForensics++_C23\test\_id_map.csv"
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    model = build_model().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Accumulate frame probabilities per video, then average -> video score
    video_probs = defaultdict(list)
    video_label = {}

    for images, labels, video_ids in test_loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

        for prob, label, video_id in zip(probs, labels, video_ids):
            video_probs[video_id].append(float(prob))
            if video_id not in video_label:
                video_label[video_id] = int(label)

    y_true, y_score, video_ids = aggregate_video_scores_mean(video_probs, video_label)

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        include_curves=True,
    )

    print("\nVIDEO-LEVEL RESULTS (FF++ C23 baseline)")
    print("Num videos:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)

    txt_path, json_path = save_metrics_report(
        out_dir=out_dir,
        name="video_level",
        metrics=metrics,
        extra={
            "dataset": "FaceForensics++ C23",
            "aggregation": "mean(frame_probs)",
            "model_path": str(model_path),
        },
    )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()
