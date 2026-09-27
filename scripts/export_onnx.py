"""
Export VerifAI models for in-browser inference (ONNX Runtime Web)
=================================================================

Produces the files served by the website (website/models/):

    visual_xception.onnx  input  frames [N,3,224,224] (ImageNet-normalised RGB)
                          output logits [N]  (224 -> 299 bilinear resize is
                          baked in, matching inference_service._score_*)
    audio_resnet18.onnx   input  mfcc   [1,1,40,200] (per-sample standardised)
                          output logit  [1]
    fusion.json           logistic-regression fusion weights + thresholds

Weights are stored as int8 (weight-only, per-channel) to shrink the download;
scripts/check_onnx_parity.py compares them with the PyTorch originals.

Usage (repo root):
    python scripts/export_onnx.py

Author: Giulio Dajani
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary  # noqa: E402
from ml.models.video.xception import build_xception_binary  # noqa: E402

OUT = ROOT / "website" / "models"
VISUAL_CKPT = ROOT / "experiments/results/final_visual_all_datasets_xception/best_model.pt"
AUDIO_CKPT = ROOT / "experiments/results/fakeavceleb_audio_resnet_baseline/best_model.pt"
FUSION_MODEL = ROOT / "experiments/results/final_hybrid_av/fusion_model.joblib"
FUSION_META = ROOT / "experiments/results/final_hybrid_av/fusion_meta.json"
VISUAL_ONLY_THRESHOLD = 0.5  # same as backend/app/services/inference_service.py


class VisualWrapper(nn.Module):
    """224x224 frames -> resize to 299 -> Xception -> one logit per frame."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)
        return self.model(x).view(-1)


class AudioWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        return self.model(mfcc).view(-1)


def _load(model: nn.Module, ckpt: Path) -> nn.Module:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    return model.eval()


def _export(module: nn.Module, example: torch.Tensor, path: Path, in_name: str, out_name: str, dynamic_batch: bool):
    dyn = {in_name: {0: "n"}, out_name: {0: "n"}} if dynamic_batch else None
    torch.onnx.export(
        module, (example,), str(path),
        input_names=[in_name], output_names=[out_name],
        dynamic_axes=dyn, opset_version=17, dynamo=False,
    )


def _quantize(src: Path, dst: Path) -> None:
    """Weight-only int8 compression (per-output-channel, symmetric).

    Conv/Gemm weights are stored as int8 + a DequantizeLinear node, so the
    download is ~4x smaller but all arithmetic still runs in float32
    (ONNX Runtime folds the dequantisation into fp32 weights at load time).
    Activations are NOT quantised, which keeps browser results close to the
    PyTorch server results.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(src))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    targets = set()
    for node in g.node:
        if node.op_type in ("Conv", "Gemm") and len(node.input) > 1 and node.input[1] in inits:
            if node.op_type == "Gemm":
                tb = next((a.i for a in node.attribute if a.name == "transB"), 0)
                if not tb:
                    continue
            targets.add(node.input[1])

    new_nodes = []
    for name in sorted(targets):
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        flat = w.reshape(w.shape[0], -1)
        scale = np.abs(flat).max(axis=1) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
        g.initializer.remove(inits[name])
        g.initializer.extend([
            numpy_helper.from_array(q, name + "_q"),
            numpy_helper.from_array(scale.astype(np.float32), name + "_scale"),
            numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + "_zp"),
        ])
        new_nodes.append(helper.make_node(
            "DequantizeLinear", [name + "_q", name + "_scale", name + "_zp"], [name], axis=0,
            name=name + "_dq"))
    for n in reversed(new_nodes):
        g.node.insert(0, n)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_fp32"
    tmp.mkdir(exist_ok=True)

    visual = VisualWrapper(_load(build_xception_binary(pretrained=False), VISUAL_CKPT))
    audio = AudioWrapper(_load(build_mfcc_resnet18_binary(pretrained=False), AUDIO_CKPT))

    with torch.no_grad():
        _export(visual, torch.randn(5, 3, 224, 224), tmp / "visual_xception.onnx", "frames", "logits", True)
        _export(audio, torch.randn(1, 1, 40, 200), tmp / "audio_resnet18.onnx", "mfcc", "logit", False)

    for name in ("visual_xception.onnx", "audio_resnet18.onnx"):
        _quantize(tmp / name, OUT / name)
        print(f"{name}: fp32 {(tmp / name).stat().st_size / 1e6:.1f} MB -> int8 {(OUT / name).stat().st_size / 1e6:.1f} MB")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    lr = joblib.load(FUSION_MODEL)
    meta = json.loads(FUSION_META.read_text(encoding="utf-8"))
    fusion = {
        "features": ["visual_prob", "audio_prob"],
        "coef": [float(c) for c in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "hybrid_threshold": float(meta.get("best_threshold", 0.5)),
        "visual_only_threshold": VISUAL_ONLY_THRESHOLD,
        "n_frames": 5,
        "image_size": 224,
        "n_mfcc": 40,
        "mfcc_max_len": 200,
        "sample_rate": 16000,
    }
    (OUT / "fusion.json").write_text(json.dumps(fusion, indent=2), encoding="utf-8")
    print("fusion.json:", fusion)


if __name__ == "__main__":
    main()
