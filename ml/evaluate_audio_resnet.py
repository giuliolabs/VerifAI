"""
Evaluate Audio-Only MFCC->ResNet model.

Outputs:
- Accuracy, F1, AUC
- Confusion matrix
- Classification report
- Saved report: experiments/results/fakeavceleb_audio_resnet_baseline/test_report.txt

Dependencies:
    pip install torch torchvision
    pip install numpy pandas scikit-learn tqdm
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from ml.audio_data_loader import MFCCDataset
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary


def main():
    test_csv = "data/splits/fakeavceleb_test.csv"
    mfcc_test_root = "data/processed/audio_features/FakeAVCeleb_v1.2/test"

    ckpt_path = Path("experiments/results/fakeavceleb_audio_resnet_baseline/best_model.pt")
    out_dir = ckpt_path.parent
    out_report = out_dir / "test_report.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = MFCCDataset(test_csv, mfcc_test_root)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    model = build_mfcc_resnet18_binary(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(y.numpy().astype(int).tolist())
            y_prob.extend(probs.tolist())

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
    report = classification_report(y_true, y_pred, digits=4)

    text = ["AUDIO-ONLY RESULTS (FakeAVCeleb MFCC->ResNet18 baseline)\n", f"Num samples: {len(y_true)}",
            f"Accuracy: {acc:.6f}", f"F1: {f1:.6f}", f"AUC: {auc:.6f}\n", "Confusion matrix:", str(cm),
            "\nClassification report:\n" + report]

    out_report.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print("\nSaved report ->", out_report)


if __name__ == "__main__":
    main()
