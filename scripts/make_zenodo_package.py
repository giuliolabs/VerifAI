"""
Build the Zenodo release of the VerifAI trained model weights
=============================================================

Creates  zenodo_release/VerifAI-model-weights-v1.0/  and a matching .zip:

    README.md               model card (what each file is, how to load it, results, licence)
    SHA256SUMS.txt          checksums for every file (integrity / reproducibility)
    LICENSE.txt             licence for the weights (CC BY-NC 4.0 by default)
    load_example.py         minimal loading + inference example
    checkpoints/<run>/...   PyTorch checkpoints + their test metrics/reports
    final_hybrid_av/...     logistic late-fusion model + threshold metadata
    onnx/...                int8-weight ONNX exports used by https://vrifai.com

Only the Python standard library is needed. Run from the repository root:

    python scripts/make_zenodo_package.py

Author: Giulio Dajani
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

VERSION = "1.0"
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
OUT_ROOT = ROOT / "zenodo_release"
NAME = f"VerifAI-model-weights-v{VERSION}"
OUT = OUT_ROOT / NAME

# (run folder, role in the paper, architecture / builder, input, training data)
RUNS = [
    ("final_visual_all_datasets_xception", "FINAL visual model (deployed)",
     "Xception (timm `legacy_xception`, 1 logit) — `ml.models.video.xception.build_xception_binary`",
     "RGB frames 299×299, ImageNet-normalised (frames are sampled at 224×224 and bilinearly resized)",
     "FaceForensics++ C23 + Celeb-DF v2 + DeeperForensics-1.0 + FakeAVCeleb v1.2 (frames)"),
    ("fakeavceleb_audio_resnet_baseline", "FINAL audio model (deployed)",
     "ResNet-18, 1-channel input, 1 logit — `ml.models.audio.mfcc_cnn.build_mfcc_resnet18_binary`",
     "MFCC [1, 40, 200] (librosa, 16 kHz, per-sample standardised)",
     "FakeAVCeleb v1.2 (audio)"),
    ("fakeavceleb_av_fusion_v1", "Audio-visual fusion v1 (ablation)",
     "`ml.models.fusion.multimodal_fusion.MultimodalFusionModel` (MobileNetV2 video + ResNet-18 MFCC, MLP head)",
     "5 frames 224×224 + MFCC [1, 40, 200]",
     "FakeAVCeleb v1.2"),
    ("ffpp_c23_xception_baseline", "Visual baseline (used in the cross-dataset / OOD evaluation)",
     "Xception — `ml.models.video.xception.build_xception_binary`",
     "RGB frames 299×299",
     "FaceForensics++ C23"),
    ("ffpp_c23_mobilenet_baseline", "Visual baseline",
     "MobileNetV2 — `ml.models.video.mobilenet_baseline.build_mobilenet_v2_binary`",
     "RGB frames 224×224",
     "FaceForensics++ C23"),
    ("fakeavceleb_wav_encoder_baseline", "Raw-waveform audio baseline",
     "1-D conv encoder — `ml.models.audio.wav_encoder.build_wav_binary_classifier`",
     "Raw waveform, 16 kHz",
     "FakeAVCeleb v1.2 (audio)"),
]
FUSION_DIR = "final_hybrid_av"
KEEP_EXT = {".json", ".txt", ".csv"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_metrics(run_dir: Path) -> str:
    """Short human-readable metric summary from test_metrics.json / reports."""
    keys = [("accuracy", "Acc"), ("balanced_accuracy", "BalAcc"), ("auc_roc", "AUC"), ("f1", "F1"), ("mcc", "MCC")]
    m = run_dir / "test_metrics.json"
    if m.exists():
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
            parts = [f"{lab} {d[k]:.3f}" for k, lab in keys if isinstance(d.get(k), (int, float))]
            if d.get("num_samples"):
                parts.append(f"n={d['num_samples']}")
            if parts:
                return ", ".join(parts)
        except Exception:
            pass
    r = run_dir / "test_report.txt"
    if r.exists():  # some reports embed a JSON block with the overall metrics
        txt = r.read_text(encoding="utf-8", errors="ignore")
        start = txt.find("{")
        if start >= 0:
            try:
                d, _ = json.JSONDecoder().raw_decode(txt[start:])
                parts = [f"{lab} {d[k]:.3f}" for k, lab in keys if isinstance(d.get(k), (int, float))]
                if d.get("num_samples"):
                    parts.append(f"n={d['num_samples']}")
                if parts:
                    return ", ".join(parts) + " (all test sets; per-dataset results in test_report.txt)"
            except ValueError:
                pass
    for rep in ("video_level_report.txt", "test_report.txt"):
        r = run_dir / rep
        if r.exists():
            vals = {}
            for line in r.read_text(encoding="utf-8", errors="ignore").splitlines():
                for key, lab in (("Accuracy:", "Acc"), ("AUC:", "AUC"), ("ROC-AUC:", "AUC"), ("F1:", "F1")):
                    if line.strip().startswith(key):
                        try:
                            vals[lab] = float(line.split(":", 1)[1])
                        except ValueError:
                            pass
            if vals:
                return ", ".join(f"{k} {v:.3f}" for k, v in vals.items()) + f" (see {rep})"
    return "see report files"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows, missing = [], []

    for run, role, arch, inp, data in RUNS:
        src = RES / run
        ckpt = src / "best_model.pt"
        if not ckpt.exists():
            missing.append(run)
            continue
        dst = OUT / "checkpoints" / run
        dst.mkdir(parents=True)
        shutil.copy2(ckpt, dst / "best_model.pt")
        for f in src.iterdir():
            if f.suffix in KEEP_EXT:
                shutil.copy2(f, dst / f.name)
        rows.append((run, role, arch, inp, data, pick_metrics(src), ckpt.stat().st_size))

    fdst = OUT / FUSION_DIR
    fdst.mkdir()
    for f in (RES / FUSION_DIR).iterdir():
        shutil.copy2(f, fdst / f.name)

    onnx_src = ROOT / "website" / "models"
    if onnx_src.exists():
        shutil.copytree(onnx_src, OUT / "onnx", ignore=shutil.ignore_patterns("_fp32"))

    hybrid = json.loads((RES / FUSION_DIR / "test_metrics.json").read_text(encoding="utf-8"))
    (OUT / "README.md").write_text(model_card(rows, hybrid, missing), encoding="utf-8")
    (OUT / "LICENSE.txt").write_text(LICENSE_TEXT, encoding="utf-8")
    (OUT / "load_example.py").write_text(LOAD_EXAMPLE, encoding="utf-8")

    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    lines = [f"{sha256(p)}  {p.relative_to(OUT).as_posix()}" for p in files]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zpath = OUT_ROOT / f"{NAME}.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, f"{NAME}/{p.relative_to(OUT).as_posix()}")
    # README + checksums also uploaded separately so Zenodo previews them
    shutil.copy2(OUT / "README.md", OUT_ROOT / "README.md")
    shutil.copy2(OUT / "SHA256SUMS.txt", OUT_ROOT / "SHA256SUMS.txt")

    total = sum(p.stat().st_size for p in files)
    print(f"Package: {OUT}  ({len(files)} files, {total / 1e6:.0f} MB)")
    print(f"Zip:     {zpath}  ({zpath.stat().st_size / 1e6:.0f} MB)")
    print(f"Also:    {OUT_ROOT / 'README.md'}, {OUT_ROOT / 'SHA256SUMS.txt'}")
    if missing:
        print("\nWARNING — checkpoints not found (not included):", ", ".join(missing))


def model_card(rows, hybrid, missing) -> str:
    table = "\n".join(
        f"| `checkpoints/{r}/best_model.pt` | {role} | {arch} | {inp} | {data} | {met} | {size / 1e6:.0f} MB |"
        for r, role, arch, inp, data, met, size in rows)
    miss = ""
    if missing:
        miss = ("\n> **Not included in this version:** " + ", ".join(f"`{m}`" for m in missing) + ".\n")
    return f"""# VerifAI — trained model weights (v{VERSION})

Trained weights for **VerifAI**, a multimodal (audio + visual) deepfake video detection framework.

- **Code:** https://github.com/giuliolabs/VerifAI
- **Live demo (runs in the browser):** https://vrifai.com
- **Paper:** Dajani, G. & Abdul Kareem, R. S. *VerifAI* (manuscript under review — citation to be added)

## Contents

| File | Role | Architecture / builder in the code | Input | Trained on | Test results | Size |
|---|---|---|---|---|---|---|
{table}
| `final_hybrid_av/fusion_model.joblib` | FINAL late-fusion layer (deployed) | scikit-learn `LogisticRegression` on `[p_visual, p_audio]` | two probabilities | FakeAVCeleb v1.2 (validation split) | Acc {hybrid.get('accuracy', 0):.3f}, BalAcc {hybrid.get('balanced_accuracy', 0):.3f}, AUC {hybrid.get('auc_roc', 0):.3f}, MCC {hybrid.get('mcc', 0):.3f}, n={hybrid.get('num_samples', '?')} (threshold {hybrid.get('threshold', 0.25)}) | <1 MB |
| `onnx/visual_xception.onnx`, `onnx/audio_resnet18.onnx`, `onnx/fusion.json` | Browser models used by vrifai.com | ONNX (opset 17) exports of the two FINAL checkpoints, int8 weight-only compression | as above | — | label agreement with PyTorch 10/10 on the parity set (`scripts/check_onnx_parity.py`) | 32 MB |
{miss}
Each checkpoint folder also contains the test metrics / reports produced by the evaluation scripts in the repository.

## Final (deployed) system

1. Sample 5 evenly spaced frames → 224×224 → resize to 299×299 → Xception → per-frame logits.
2. Extract audio → 16 kHz → 40 MFCC × 200 frames → per-sample standardisation → ResNet-18 → logit.
3. If usable audio exists: `p = LogisticRegression([mean(sigmoid(visual_logits)), sigmoid(audio_logit)])`, label *fake* if `p ≥ 0.25`.
   Otherwise (no audio): `p = sigmoid(mean(visual_logits))`, label *fake* if `p ≥ 0.5`.

Reference implementation: `backend/app/services/inference_service.py` and `preprocess_service.py`.

## Loading

```python
import torch
from ml.models.video.xception import build_xception_binary          # from the GitHub repository

model = build_xception_binary(pretrained=False)
ckpt = torch.load("checkpoints/final_visual_all_datasets_xception/best_model.pt",
                  map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
```

Each `.pt` file is a dict with `model_state` (the `state_dict`) plus training metadata (epoch, validation metrics, config).
See `load_example.py` for a complete example that runs the full hybrid pipeline on a video.

Environment used: Python 3.11–3.12, PyTorch 2.x, torchvision, timm, librosa 0.11, scikit-learn (fusion model pickled with 1.5.2).

## Integrity

Verify downloads with `sha256sum -c SHA256SUMS.txt` (Linux/macOS) or `Get-FileHash` (Windows).

## Training data and intended use

The models were trained on public research datasets — FaceForensics++, Celeb-DF v2, DeeperForensics-1.0 and FakeAVCeleb v1.2 —
each released under its own research-only / non-commercial terms. **No dataset videos are redistributed here**;
obtain them from their original authors. The cross-dataset evaluation set (Survey369) is not redistributed.

These weights are released for **research and educational use**. They are a research prototype: benchmark-trained detectors
generalise poorly to out-of-distribution, in-the-wild video (a central finding of the accompanying paper), so scores must not be
used as sole evidence for forensic, legal or journalistic decisions.

## Licence

Weights: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** — see `LICENSE.txt`.
Code: see the GitHub repository.

## Citation

If you use these weights, please cite this Zenodo record (DOI on the record page) and the paper.
"""


LICENSE_TEXT = """VerifAI trained model weights
Copyright (c) 2026 Giulio Dajani

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
You may share and adapt these files for non-commercial purposes, provided you give appropriate credit,
provide a link to the licence, and indicate if changes were made.

Full licence text: https://creativecommons.org/licenses/by-nc/4.0/legalcode

The models were trained on third-party research datasets (FaceForensics++, Celeb-DF v2, DeeperForensics-1.0,
FakeAVCeleb v1.2); users are responsible for complying with those datasets' terms of use.
"""

LOAD_EXAMPLE = '''"""
Run the final VerifAI hybrid detector on one video with these weights.

Usage (inside a clone of https://github.com/giuliolabs/VerifAI, with its requirements installed):
    python load_example.py path/to/weights_folder path/to/video.mp4
"""
import json, math, sys
from pathlib import Path

import joblib, torch, torch.nn.functional as F

from ml.models.video.xception import build_xception_binary
from ml.models.audio.mfcc_cnn import build_mfcc_resnet18_binary
from backend.app.services.preprocess_service import extract_features_from_video

w, video_path = Path(sys.argv[1]), sys.argv[2]

def load(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state"])
    return model.eval()

visual = load(build_xception_binary(pretrained=False), w / "checkpoints/final_visual_all_datasets_xception/best_model.pt")
audio = load(build_mfcc_resnet18_binary(pretrained=False), w / "checkpoints/fakeavceleb_audio_resnet_baseline/best_model.pt")
fusion = joblib.load(w / "final_hybrid_av/fusion_model.joblib")
threshold = json.loads((w / "final_hybrid_av/fusion_meta.json").read_text())["best_threshold"]

video, mfcc = extract_features_from_video(video_path)            # [5,3,224,224], [1,40,200]
with torch.no_grad():
    logits = visual(F.interpolate(video, size=(299, 299), mode="bilinear", align_corners=False)).view(-1)
    if torch.all(mfcc == 0):                                        # no usable audio -> visual only
        p, mode, thr = torch.sigmoid(logits.mean()).item(), "visual_only", 0.5
    else:
        pv = torch.sigmoid(logits).mean().item()
        pa = torch.sigmoid(audio(mfcc.unsqueeze(0)).view(-1)[0]).item()
        p, mode, thr = float(fusion.predict_proba([[pv, pa]])[0, 1]), "hybrid_av", threshold
print({"label": "fake" if p >= thr else "real", "prob_fake": round(p, 4), "mode": mode})
'''

if __name__ == "__main__":
    main()
