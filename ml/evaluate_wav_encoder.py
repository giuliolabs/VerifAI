"""
Evaluate Audio-Only WAV->1D CNN encoder model.

Outputs:
- Accuracy, F1, AUC
- Confusion matrix
- Classification report
- Saved report:
    experiments/results/fakeavceleb_wav_encoder_baseline/test_report.txt

Dependencies:
    pip install torch numpy pandas scikit-learn tqdm soundfile
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from ml.wav_data_loader import WavDataset
from ml.models.audio.wav_encoder import build_wav_binary_classifier


def _torch_load_compat(path: Path, device: str):
    """
    PyTorch 2.6+ changed torch.load default weights_only=True.
    Our checkpoint is trusted (created by this project), so load with weights_only=False when available.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    test_csv = "data/splits/fakeavceleb_test.csv"
    wav_test_root = "data/interim/audio/FakeAVCeleb_v1.2/test"

    ckpt_path = Path("experiments/results/fakeavceleb_wav_encoder_baseline/best_model.pt")
    out_dir = ckpt_path.parent
    out_report = out_dir / "test_report.txt"

    sample_rate = 16000
    seconds = 3
    strict_sr = False  # set True if you know all wav files are 16kHz

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = WavDataset(
        split_csv=test_csv,
        wav_root=wav_test_root,
        sample_rate=sample_rate,
        seconds=seconds,
        strict_sr=strict_sr,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

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
    y_prob: list[float] = []

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Evaluating"):
            x = x.to(device)                 # [B,1,T]
            logits = model(x).squeeze(1)     # [B]
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(y.numpy().astype(int).tolist())
            y_prob.extend(probs.tolist())

    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    y_pred_np = (y_prob_np >= 0.5).astype(int)

    bal_acc = balanced_accuracy_score(y_true_np, y_pred_np)
    mcc = matthews_corrcoef(y_true_np, y_pred_np)

    preds_csv = out_dir / "predictions.csv"
    with preds_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "prob_fake", "pred"])
        for i, (yt, yp, yhat) in enumerate(zip(y_true_np.tolist(), y_prob_np.tolist(), y_pred_np.tolist())):
            writer.writerow([i, yt, f"{yp:.6f}", yhat])

    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np)

    try:
        auc = roc_auc_score(y_true_np, y_prob_np)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true_np, y_pred_np)
    report = classification_report(y_true_np, y_pred_np, digits=4)

    text = [
        "AUDIO-ONLY RESULTS (FakeAVCeleb WAV->1D CNN baseline)\n",
        f"Num samples: {len(y_true_np)}",
        f"Accuracy: {acc:.6f}",
        f"F1: {f1:.6f}",
        f"AUC: {auc:.6f}",
        f"Balanced Accuracy: {bal_acc:.6f}",
        f"MCC: {mcc:.6f}",
        f"Predictions CSV: {preds_csv}\n",
        "Confusion matrix:",
        str(cm),
        "\nClassification report:\n" + report,
    ]

    out_report.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print("\nSaved report ->", out_report)


if __name__ == "__main__":
    main()
