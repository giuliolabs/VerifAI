"""
Train FINAL hybrid AV fusion for VerifAI.

Uses:
- visual backbone trained on all frame datasets
- audio backbone trained on FakeAVCeleb
- learns late fusion on FakeAVCeleb train
- tunes threshold on FakeAVCeleb val

Run:
    python -m ml.train_final_hybrid_av

Output:
    experiments/results/final_hybrid_av/fusion_model.joblib
    experiments/results/final_hybrid_av/fusion_meta.json
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import joblib
import numpy as np
import torch
import torch.nn.functional as f
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from ml.av_data_loader import FakeAVCelebAVDataset
from ml.models.video.xception import build_xception_binary
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary


VISUAL_CKPT = Path("experiments/results/final_visual_all_datasets_xception/best_model.pt")
AUDIO_CKPT = Path("experiments/results/fakeavceleb_audio_resnet_baseline/best_model.pt")
OUT_DIR = Path("experiments/results/final_hybrid_av")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _torch_load_compat(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def extract_features(split: str, visual_model, audio_model, device: str):
    ds = FakeAVCelebAVDataset(split, strict=True, mfcc_max_len=200)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    x_all = []
    y_all = []

    for video, mfcc, y, _ in tqdm(loader, desc=f"Extract {split}"):
        video = video.to(device)   # [B,T,3,224,224]
        mfcc = mfcc.to(device)

        batch_size, time_steps, channels, height, width = video.shape
        frames = video.view(batch_size * time_steps, channels, height, width)
        frames = f.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)

        visual_logits = visual_model(frames).view(batch_size, time_steps)
        visual_prob = torch.sigmoid(visual_logits).mean(dim=1)

        audio_logits = audio_model(mfcc).squeeze(1)
        audio_prob = torch.sigmoid(audio_logits)

        features = torch.stack([visual_prob, audio_prob], dim=1).cpu().numpy()
        x_all.append(features)
        y_all.append(y.cpu().numpy().astype(int))

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return x_all, y_all


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not VISUAL_CKPT.exists():
        raise FileNotFoundError(f"Missing visual checkpoint: {VISUAL_CKPT}")
    if not AUDIO_CKPT.exists():
        raise FileNotFoundError(f"Missing audio checkpoint: {AUDIO_CKPT}")

    visual_model = build_xception_binary(pretrained=False).to(device)
    audio_model = build_mfcc_resnet18_binary(pretrained=False).to(device)

    visual_ckpt = _torch_load_compat(VISUAL_CKPT, device)
    audio_ckpt = _torch_load_compat(AUDIO_CKPT, device)

    visual_model.load_state_dict(visual_ckpt["model_state"])
    audio_model.load_state_dict(audio_ckpt["model_state"])

    visual_model.eval()
    audio_model.eval()

    x_train, y_train = extract_features("train", visual_model, audio_model, device)
    x_val, y_val = extract_features("val", visual_model, audio_model, device)

    fusion = LogisticRegression(max_iter=1000, class_weight="balanced")
    fusion.fit(x_train, y_train)

    val_prob = fusion.predict_proba(x_val)[:, 1]

    best_t = 0.5
    best_bal = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        preds = (val_prob >= t).astype(int)
        bal = balanced_accuracy_score(y_val, preds)
        if bal > best_bal:
            best_bal = float(bal)
            best_t = float(t)

    joblib.dump(fusion, OUT_DIR / "fusion_model.joblib")

    meta = {
        "visual_checkpoint": str(VISUAL_CKPT),
        "audio_checkpoint": str(AUDIO_CKPT),
        "best_threshold": best_t,
        "best_val_balanced_accuracy": best_bal,
        "features": ["visual_prob", "audio_prob"],
        "dataset": "FakeAVCeleb",
        "model": "Final hybrid AV fusion",
    }
    (OUT_DIR / "fusion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved fusion model ->", OUT_DIR / "fusion_model.joblib")
    print("Saved metadata ->", OUT_DIR / "fusion_meta.json")
    print("Best threshold:", best_t)
    print("Best val balanced accuracy:", best_bal)


if __name__ == "__main__":
    main()
