"""
Evaluate Baseline CNN (MobileNetV2) on FaceForensics++ C23 (video-level)
Dependencies:
    pip install torch torchvision
    pip install scikit-learn tqdm pillow numpy
Run:
    python ml/evaluate.py
Output:
    experiments/results/ffpp_c23_mobilenet_baseline/test_report.txt
"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from ml.data_loader import FFPPFrameDataset
from ml.models.video.mobilenet_baseline import build_mobilenet_v2_binary


def make_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def main():
    data_root = Path("data/interim/frames/FaceForensics++_C23")
    out_dir = Path("experiments/results/ffpp_c23_mobilenet_baseline")
    model_path = out_dir / "best_model.pt"
    report_path = out_dir / "test_report.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run ml/train.py first.")

    test_dir = data_root / "test"
    test_ds = FFPPFrameDataset(str(test_dir), transform=make_transforms())
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    model = build_mobilenet_v2_binary().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

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

    y_true = []
    y_score = []

    for video_id in sorted(video_probs.keys()):
        y_true.append(video_label[video_id])
        y_score.append(float(np.mean(video_probs[video_id])))

    y_pred = [1 if p >= 0.5 else 0 for p in y_score]

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4)

    print("\nVIDEO-LEVEL RESULTS (FF++ C23 baseline)")
    print("Num videos:", len(y_true))
    print("Accuracy:", acc)
    print("AUC:", auc)
    print("Confusion matrix:\n", cm)
    print("\nReport:\n", rep)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("VIDEO-LEVEL RESULTS (FF++ C23 baseline)\n")
        f.write(f"Num videos: {len(y_true)}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"AUC: {auc}\n")
        f.write(f"Confusion matrix:\n{cm}\n\n")
        f.write(rep)

    print("\nSaved report ->", report_path)


if __name__ == "__main__":
    main()
