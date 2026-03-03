"""
Evaluate Audio-Only WAV->1D CNN encoder model (FakeAVCeleb v1.2).

Writes consistent metrics for consolidation:
- Accuracy
- Balanced Accuracy
- F1 (from sklearn report)
- ROC-AUC
- Average Precision (AP)
- MCC
- Confusion matrix + classification report
- Saves:
    experiments/results/fakeavceleb_wav_encoder_baseline/test_report.txt
    experiments/results/fakeavceleb_wav_encoder_baseline/test_metrics.json
    experiments/results/fakeavceleb_wav_encoder_baseline/predictions.csv

Run:
    python -m ml.evaluate_wav_encoder
(or whatever module name you use for this file)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.wav_data_loader import WavDataset
from ml.models.audio.wav_encoder import build_wav_binary_classifier
from ml.metrics import compute_binary_metrics, save_metrics_report


def _torch_load_compat(path: Path, device: str):
    """
    PyTorch 2.6+ changed torch.load default weights_only=True.
    Our checkpoint is trusted (created by this project), so load with weights_only=False when available.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def main() -> None:
    test_csv = "data/splits/fakeavceleb_test.csv"
    wav_test_root = "data/interim/audio/FakeAVCeleb_v1.2/test"

    ckpt_path = Path("experiments/results/fakeavceleb_wav_encoder_baseline/best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_csv = out_dir / "predictions.csv"

    sample_rate = 16000
    seconds = 3
    strict_sr = False  # True only if you KNOW all wav files are exactly 16kHz

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- data ----
    ds = WavDataset(
        split_csv=test_csv,
        wav_root=wav_test_root,
        sample_rate=sample_rate,
        seconds=seconds,
        strict_sr=strict_sr,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    # ---- model ----
    model = build_wav_binary_classifier(
        sample_rate=sample_rate,
        embedding_dim=256,
        base_channels=32,
        dropout=0.2,
    ).to(device)

    ckpt = _torch_load_compat(ckpt_path, device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    y_true: list[int] = []
    y_score: list[float] = []

    # ---- inference ----
    for x, y, _ in tqdm(loader, desc="Evaluating"):
        x = x.to(device)              # [B,1,T]
        logits = model(x).squeeze(1)  # [B]
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        y_true.extend([int(v) for v in y.cpu().numpy().tolist()])
        y_score.extend([float(p) for p in probs.tolist()])

    # ---- metrics (single source of truth) ----
    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        include_curves=True,  # enables ROC-AUC + AP reliably
    )

    # ---- save per-sample predictions ----
    y_pred = (np.array(y_score) >= 0.5).astype(int).tolist()
    with preds_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "prob_fake", "pred"])
        for i, (yt, yp, yhat) in enumerate(zip(y_true, y_score, y_pred)):
            writer.writerow([i, yt, f"{yp:.6f}", yhat])

    # ---- print + save report/json ----
    print("\nAUDIO-ONLY RESULTS (FakeAVCeleb WAV->1D CNN baseline)")
    print("Checkpoint:", ckpt_path)
    print("Num samples:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("Balanced accuracy:", getattr(metrics, "balanced_accuracy", None))
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("MCC:", getattr(metrics, "mcc", None))
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)
    print("\nPredictions CSV:", preds_csv)

    txt_path, json_path = save_metrics_report(
        out_dir=out_dir,
        name="test",
        metrics=metrics,
        extra={
            "dataset": "FakeAVCeleb v1.2",
            "model": "WAV->1D CNN baseline",
            "sample_rate": sample_rate,
            "seconds": seconds,
            "threshold": 0.5,
            "checkpoint": str(ckpt_path),
            "predictions_csv": str(preds_csv),
        },
    )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)


if __name__ == "__main__":
    main()
