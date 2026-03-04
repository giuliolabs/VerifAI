"""
Evaluate Multimodal Fusion Model (Audio + Video) on FakeAVCeleb_v1.2 (TEST split).

Outputs (in experiments/results/fakeavceleb_av_fusion_v1):
- test_report.txt
- test_metrics.json

Run (from project root):
    python -m ml.evaluate_av_fusion

Dependencies:
    pip install torch torchvision numpy scikit-learn tqdm

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.av_data_loader import FakeAVCelebAVDataset
from ml.models.fusion.multimodal_fusion import MultimodalFusionModel
from ml.metrics import compute_binary_metrics, save_metrics_report


def _to_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    # supports [B], [B,1]
    if logits.ndim == 2 and logits.size(1) == 1:
        return logits.squeeze(1)
    return logits


@torch.no_grad()
def main():
    ckpt_path = Path("experiments/results/fakeavceleb_av_fusion_v1/best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    def _torch_load_compat(path: Path, device: str):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    ckpt = _torch_load_compat(ckpt_path, device)
    mfcc_max_len = int(ckpt.get("mfcc_max_len", 200))

    # ---- data ----
    test_ds = FakeAVCelebAVDataset("test", strict=True, mfcc_max_len=mfcc_max_len)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # ---- model ----
    model = MultimodalFusionModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true: list[int] = []
    y_score: list[float] = []

    for video, mfcc, y, _ in tqdm(test_loader, desc="Evaluating"):
        video = video.to(device)
        mfcc = mfcc.to(device)

        logits = _to_1d_logits(model(video, mfcc))
        probs = torch.sigmoid(logits).cpu().numpy()

        y_score.extend([float(p) for p in probs.tolist()])
        # y might be tensor/list; normalize to int
        if isinstance(y, torch.Tensor):
            y_true.extend([int(v) for v in y.cpu().numpy().tolist()])
        else:
            y_true.extend([int(v) for v in y])

    # ---- metrics (single source of truth) ----
    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        include_curves=True,
    )

    print("\nAUDIO+VIDEO FUSION RESULTS (FakeAVCeleb AV Fusion v1)")
    print("Checkpoint:", ckpt_path)
    print("MFCC max len (batching):", mfcc_max_len)
    print("Num samples:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("Balanced accuracy:", getattr(metrics, "balanced_accuracy", None))
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("MCC:", getattr(metrics, "mcc", None))
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)

    txt_path, json_path = save_metrics_report(
        out_dir=out_dir,
        name="test",
        metrics=metrics,
        extra={
            "dataset": "FakeAVCeleb",
            "model": "AV Fusion v1",
            "mfcc_max_len": mfcc_max_len,
            "checkpoint": str(ckpt_path),
        },
    )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()
