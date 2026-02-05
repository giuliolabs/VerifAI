"""
Evaluate Multimodal Fusion Model (Audio + Video) on FakeAVCeleb_v1.2 (TEST split).

What it does
------------
- Loads FakeAVCeleb test split via ml.av_data_loader.FakeAVCelebAVDataset
- Loads checkpoint: experiments/results/fakeavceleb_av_fusion_v1/best_model.pt
- Runs inference and computes:
    - Accuracy, F1, AUC
    - Confusion matrix
    - Classification report
- Saves report to:
    experiments/results/fakeavceleb_av_fusion_v1/test_report.txt

Run (from project root)
-----------------------
python -m ml.evaluate_av_fusion

Dependencies
------------
pip install torch torchvision numpy scikit-learn tqdm

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from ml.av_data_loader import FakeAVCelebAVDataset
from ml.models.fusion.multimodal_fusion import MultimodalFusionModel


def _to_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    # supports [B], [B,1]
    if logits.ndim == 2 and logits.size(1) == 1:
        return logits.squeeze(1)
    return logits


def main():
    ckpt_path = Path("experiments/results/fakeavceleb_av_fusion_v1/best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = ckpt_path.parent
    out_report = out_dir / "test_report.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt = torch.load(ckpt_path, map_location=device)
    mfcc_max_len = ckpt.get("mfcc_max_len", 200)

    # ---- data ----
    test_ds = FakeAVCelebAVDataset("test", strict=True, mfcc_max_len=mfcc_max_len)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # ---- model ----
    model = MultimodalFusionModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for video, mfcc, y, _ in tqdm(test_loader, desc="Evaluating"):
            video = video.to(device)
            mfcc = mfcc.to(device)

            logits = model(video, mfcc)
            logits = _to_1d_logits(logits)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_prob.extend(probs.tolist())

            y_true.extend(y.detach().cpu().numpy().astype(int).tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4)

    text = [
        "AUDIO+VIDEO FUSION RESULTS (FakeAVCeleb AV Fusion v1)\n",
        f"Checkpoint: {ckpt_path}",
        f"MFCC max len (batching): {mfcc_max_len}",
        f"Num samples: {len(y_true)}",
        "",
        f"Accuracy: {acc:.6f}",
        f"F1: {f1:.6f}",
        f"AUC: {auc:.6f}",
        "",
        "Confusion matrix:",
        str(cm),
        "",
        "Classification report:",
        rep,
    ]

    out_report.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print("\nSaved report ->", out_report)


if __name__ == "__main__":
    main()
