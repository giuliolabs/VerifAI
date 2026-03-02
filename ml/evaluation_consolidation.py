"""
Evaluation Consolidation Script (Final)

Scans experiments/results/* for model reports and metrics JSON,
extracts key metrics, and builds:

- experiments/results/_summary/model_comparison.csv
- experiments/results/_summary/model_comparison.md

Run:
    python -m ml.evaluation_consolidation
"""

from __future__ import annotations

import re
import json
import csv
from pathlib import Path
from typing import Optional, Dict, Tuple


RESULTS_ROOT = Path("experiments/results")
SUMMARY_DIR = RESULTS_ROOT / "_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = SUMMARY_DIR / "model_comparison.csv"
MD_PATH = SUMMARY_DIR / "model_comparison.md"


# ---------- helpers ----------
def fmt(value: Optional[float], ndigits: int = 4) -> str:
    """Safe float formatting."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return "N/A"


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def detect_dataset(model_name: str, report_text: str) -> str:
    t = (model_name + "\n" + report_text).lower()

    if "faceforensics" in t or "ff++" in t or "ffpp" in t:
        if "c23" in t:
            return "FaceForensics++ C23"
        if "c40" in t:
            return "FaceForensics++ C40"
        return "FaceForensics++"

    if "celeb" in t and "df" in t:
        return "Celeb-DF v2"

    if "deeperforensics" in t:
        return "DeeperForensics"

    if "fakeavceleb" in t:
        return "FakeAVCeleb"

    # fallback: guess from folder name
    mn = model_name.lower()
    if "celebdf" in mn:
        return "Celeb-DF v2"
    if "deeper" in mn:
        return "DeeperForensics"
    if "fakeav" in mn:
        return "FakeAVCeleb"
    if "ffpp" in mn or "faceforensics" in mn:
        return "FaceForensics++"

    return "Unknown"


def detect_model_type(model_name: str, report_text: str) -> str:
    """
    Strict model type detection.
    Order matters.
    """

    t = (model_name + " " + report_text).lower()
    mn = model_name.lower()

    # Explicit fusion models only
    if "fusion" in mn or "multimodal" in mn:
        return "AV fusion"

    # Visual models
    visual_backbones = ["xception", "mobilenet", "vit", "vision", "temporal"]
    if any(v in mn for v in visual_backbones):
        return "Visual"

    # Audio models
    audio_keywords = ["mfcc", "wav", "audio", "mel", "spectrogram"]
    if any(a in mn for a in audio_keywords):
        return "Audio"

    return "Unknown"


def detect_backbone(model_name: str, report_text: str) -> str:
    t = (model_name + " " + report_text).lower()

    if "xception" in t:
        return "Xception"
    if "mobilenet" in t or "mobilenetv2" in t:
        return "MobileNetV2"
    if "vit" in t or "vision transformer" in t:
        return "ViT"
    if "temporal" in t and "vit" in t:
        return "Temporal ViT"
    if "temporal" in t and "mobilenet" in t:
        return "Temporal MobileNetV2"

    if "resnet" in t:
        m = re.search(r"resnet\s*([0-9]{2,3})", t)
        return f"ResNet{m.group(1)}" if m else "ResNet"

    if "mfcc" in t and ("cnn" in t or "conv" in t):
        return "MFCC-CNN"

    if "wav_encoder" in t or "wav encoder" in t or "wave encoder" in t:
        return "WAV encoder"

    if "efficientnet" in t:
        return "EfficientNet"

    if "wav2vec" in t:
        return "Wav2Vec"
    if "hubert" in t:
        return "HuBERT"

    return "Unknown"


def find_report_and_metrics_json(folder: Path) -> Tuple[Optional[Path], Optional[Path]]:
    report_candidates = [
        folder / "test_report.txt",
        folder / "video_level_report.txt",
        folder / "report.txt",
    ]
    json_candidates = [
        folder / "test_metrics.json",
        folder / "video_level_metrics.json",
        folder / "metrics.json",
    ]

    report_path = next((p for p in report_candidates if p.exists()), None)
    json_path = next((p for p in json_candidates if p.exists()), None)

    if report_path is None:
        txts = sorted(folder.glob("**/*report*.txt"))
        report_path = txts[0] if txts else None

    if json_path is None:
        js = sorted(folder.glob("**/*metrics*.json"))
        json_path = js[0] if js else None

    return report_path, json_path


def parse_from_json(json_path: Path) -> Dict[str, Optional[float]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    def pick(*keys):
        for k in keys:
            if k in data:
                return safe_float(data[k])
        return None

    return {
        "accuracy": pick("accuracy", "acc"),
        "auc": pick("auc_roc", "roc_auc", "auc"),
        "ap": pick("ap", "average_precision"),
        "balanced_accuracy": pick("balanced_accuracy", "bal_acc", "balanced_acc"),
        "mcc": pick("mcc", "matthews_corrcoef", "matthews_correlation"),
        "f1_weighted": pick("f1_weighted", "weighted_f1"),
        "f1_macro": pick("f1_macro", "macro_f1"),
    }


def parse_from_text(report_text: str) -> Dict[str, Optional[float]]:
    patterns = {
        "accuracy": r"Accuracy:\s*([0-9.]+)",
        "auc": r"ROC-AUC:\s*([0-9.]+)",
        "ap": r"Average precision:\s*([0-9.]+)",
        "f1_weighted": r"weighted avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)",
        "f1_macro": r"macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)",
    }

    out: Dict[str, Optional[float]] = {}
    for key, pat in patterns.items():
        m = re.search(pat, report_text)
        out[key] = safe_float(m.group(1)) if m else None

    m_bal = re.search(r"Balanced\s*Acc(?:uracy)?:\s*([0-9.]+)", report_text, re.IGNORECASE)
    out["balanced_accuracy"] = safe_float(m_bal.group(1)) if m_bal else None

    m_mcc = re.search(r"\bMCC\b\s*:\s*([0-9.\-]+)", report_text, re.IGNORECASE)
    out["mcc"] = safe_float(m_mcc.group(1)) if m_mcc else None

    return out


def merge_metrics(primary: Dict[str, Optional[float]], fallback: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    merged = dict(primary)
    for k, v in fallback.items():
        if merged.get(k) is None and v is not None:
            merged[k] = v
    return merged


def discover_models():
    models = []
    for folder in sorted(RESULTS_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name == "_summary":
            continue
        # optional skip: folders that are not experiments
        if folder.name.lower() in {"preprocessing_checks"}:
            continue
        models.append(folder)
    return models


def write_csv(rows):
    headers = [
        "Model",
        "Dataset",
        "Model type",
        "Backbone",
        "Accuracy",
        "Balanced Acc",
        "F1 (weighted)",
        "F1 (macro)",
        "ROC-AUC",
        "AP",
        "MCC",
        "Report path",
        "Metrics json path",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(r)


def write_markdown(rows):
    lines = [
        "# Model Comparison Summary\n",
        "Auto-generated from `experiments/results/*`.\n",
        "| Model | Dataset | Type | Backbone | Acc | Bal Acc | F1(w) | F1(m) | AUC | AP | MCC |",
        "|------|---------|------|----------|-----|---------|-------|-------|-----|----|-----|",
    ]

    for r in rows:
        (
            model, dataset, model_type, backbone,
            acc, bal, f1w, f1m, auc, ap, mcc,
            report_path, json_path
        ) = r

        lines.append(
            f"| {model} | {dataset} | {model_type} | {backbone} | "
            f"{fmt(acc)} | {fmt(bal)} | {fmt(f1w)} | {fmt(f1m)} | {fmt(auc)} | {fmt(ap)} | {fmt(mcc)} |"
        )

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    model_folders = discover_models()
    if not model_folders:
        print("No model folders found under experiments/results/")
        return

    rows = []

    for folder in model_folders:
        model_name = folder.name
        report_path, json_path = find_report_and_metrics_json(folder)

        if report_path is None and json_path is None:
            continue

        report_text = ""
        if report_path and report_path.exists():
            report_text = report_path.read_text(encoding="utf-8", errors="ignore")

        metrics_from_json = {}
        if json_path and json_path.exists():
            try:
                metrics_from_json = parse_from_json(json_path)
            except Exception:
                metrics_from_json = {}

        metrics_from_text = parse_from_text(report_text)
        metrics = merge_metrics(metrics_from_json, metrics_from_text)

        dataset = detect_dataset(model_name, report_text)
        model_type = detect_model_type(model_name, report_text)
        backbone = detect_backbone(model_name, report_text)

        rows.append((
            model_name,
            dataset,
            model_type,
            backbone,
            metrics.get("accuracy"),
            metrics.get("balanced_accuracy"),
            metrics.get("f1_weighted"),
            metrics.get("f1_macro"),
            metrics.get("auc"),
            metrics.get("ap"),
            metrics.get("mcc"),
            str(report_path) if report_path else "",
            str(json_path) if json_path else "",
        ))

    if not rows:
        print("No usable reports/metrics found.")
        return

    write_csv(rows)
    write_markdown(rows)

    print("Summary CSV ->", CSV_PATH)
    print("Summary MD  ->", MD_PATH)


if __name__ == "__main__":
    main()
