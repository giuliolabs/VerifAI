"""
Evaluate Visual CNN (Xception) on FaceForensics++ C23 (video-level metrics).

Dependencies:
    pip install torch torchvision timm
    pip install scikit-learn tqdm pillow numpy

Run (from project root):
    python -m ml.evaluate_xception

Outputs:
    experiments/results/ffpp_c23_xception_baseline/test_report.txt
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from ml.data_loader import FFPPFrameDataset
from ml.models.video.xception import build_xception_binary


def make_transforms():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main():
    data_root = Path("data/interim/frames/FaceForensics++_C23")
    test_dir = data_root / "test"

    ckpt_path = Path("experiments/results/ffpp_c23_xception_baseline/best_model.pt")
    out_dir = Path("experiments/results/ffpp_c23_xception_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ds = FFPPFrameDataset(str(test_dir), transform=make_transforms())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_xception_binary(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # frame -> probability, then aggregate per video by mean(prob)
    video_probs = {}
    video_labels = {}

    with torch.no_grad():
        for images, labels, video_ids in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            for p, y, vid in zip(probs, labels, video_ids):
                vid = str(vid)
                y = int(y)
                video_probs.setdefault(vid, []).append(float(p))
                video_labels[vid] = y

    vids = sorted(video_probs.keys())
    y_true = np.array([video_labels[v] for v in vids])
    y_score = np.array([np.mean(video_probs[v]) for v in vids])
    y_pred = (y_score >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print("\nVIDEO-LEVEL RESULTS (FF++ C23 Xception baseline)")
    print("Num videos:", len(vids))
    print("Accuracy:", acc)
    print("F1:", f1)
    print("AUC:", auc)
    print("Confusion matrix:\n", cm)
    print("\nReport:\n", report)

    report_path = out_dir / "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("VIDEO-LEVEL RESULTS (FF++ C23 Xception baseline)\n")
        f.write(f"Num videos: {len(vids)}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"AUC: {auc}\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n")

    print("\nSaved report ->", report_path)


if __name__ == "__main__":
    main()
