"""
Parity check: PyTorch (server) vs ONNX int8 (browser) models
============================================================

Runs the backend preprocessing on each video, then scores the same tensors
with the original PyTorch checkpoints and with website/models/*.onnx, and
prints both results side by side. Writes parity_report.json.

Usage (repo root):
    python scripts/check_onnx_parity.py path/to/clip1.mp4 path/to/clip2.mp4 ...

Author: Giulio Dajani
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.preprocess_service import extract_features_from_video  # noqa: E402
from scripts.export_onnx import AUDIO_CKPT, VISUAL_CKPT, _load  # noqa: E402
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary  # noqa: E402
from ml.models.video.xception import build_xception_binary  # noqa: E402

MODELS = ROOT / "website" / "models"
FUSION = json.loads((MODELS / "fusion.json").read_text())


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def decide(visual_logits: np.ndarray, audio_logit: float | None) -> dict:
    """Same decision logic as backend/app/services/inference_service.py."""
    if audio_logit is None:
        p = sigmoid(float(visual_logits.mean()))
        return {"mode": "visual_only_fallback", "prob_fake": p,
                "label": "fake" if p >= FUSION["visual_only_threshold"] else "real"}
    pv = float(np.mean(1 / (1 + np.exp(-visual_logits))))
    pa = sigmoid(audio_logit)
    z = FUSION["coef"][0] * pv + FUSION["coef"][1] * pa + FUSION["intercept"]
    p = sigmoid(z)
    return {"mode": "hybrid_av", "prob_fake": p, "p_visual": pv, "p_audio": pa,
            "label": "fake" if p >= FUSION["hybrid_threshold"] else "real"}


def main(paths: list[str]) -> None:
    vis_pt = _load(build_xception_binary(pretrained=False), VISUAL_CKPT)
    aud_pt = _load(build_mfcc_resnet18_binary(pretrained=False), AUDIO_CKPT)
    vis_ox = ort.InferenceSession(str(MODELS / "visual_xception.onnx"), providers=["CPUExecutionProvider"])
    aud_ox = ort.InferenceSession(str(MODELS / "audio_resnet18.onnx"), providers=["CPUExecutionProvider"])

    report = []
    print(f"{'clip':32} {'mode':22} {'torch':>8} {'onnx':>8} {'|diff|':>7}  label")
    for path in paths:
        video, mfcc = extract_features_from_video(path)
        has_audio = mfcc.numel() > 0 and not torch.all(mfcc == 0)
        with torch.no_grad():
            frames = F.interpolate(video, size=(299, 299), mode="bilinear", align_corners=False)
            vl_pt = vis_pt(frames).view(-1).numpy()
            al_pt = float(aud_pt(mfcc.unsqueeze(0)).view(-1)[0]) if has_audio else None
        vl_ox = vis_ox.run(None, {"frames": video.numpy()})[0].reshape(-1)
        al_ox = float(aud_ox.run(None, {"mfcc": mfcc.unsqueeze(0).numpy()})[0].reshape(-1)[0]) if has_audio else None

        a, b = decide(vl_pt, al_pt), decide(vl_ox, al_ox)
        diff = abs(a["prob_fake"] - b["prob_fake"])
        ok = "same" if a["label"] == b["label"] else "DIFFERENT"
        print(f"{Path(path).name[:32]:32} {a['mode']:22} {a['prob_fake']*100:7.2f}% {b['prob_fake']*100:7.2f}% {diff*100:6.2f}  {a['label']}/{b['label']} {ok}")
        report.append({"clip": Path(path).name, "torch": a, "onnx": b,
                       "visual_logits_torch": vl_pt.tolist(), "visual_logits_onnx": vl_ox.tolist(),
                       "audio_logit_torch": al_pt, "audio_logit_onnx": al_ox})

    Path("parity_report.json").write_text(json.dumps(report, indent=2))
    agree = sum(r["torch"]["label"] == r["onnx"]["label"] for r in report)
    print(f"\nLabel agreement: {agree}/{len(report)}   max |Δp| = "
          f"{max(abs(r['torch']['prob_fake'] - r['onnx']['prob_fake']) for r in report) * 100:.2f} pts")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
