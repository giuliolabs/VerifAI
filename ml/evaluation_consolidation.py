"""
Evaluation Consolidation Script (Final)
=======================================

This script scans the experiment results directory and consolidates
evaluation outputs into a single comparison summary. Its purpose is
to make cross-model analysis easier by collecting metrics from multiple
reports and presenting them in a unified CSV and Markdown format.

It builds:
- experiments/results/_summary/model_comparison.csv
- experiments/results/_summary/model_comparison.md

The script is useful for:
- comparing multiple baselines and final models
- identifying the strongest model per dataset
- highlighting reliability issues such as imbalance bias
- supporting dissertation/report discussion with one summary source

Run:
    python -m ml.evaluation_consolidation

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

import re
import json
import csv
from pathlib import Path
from typing import Optional, Dict, Tuple


# Root results directory containing all experiment folders
RESULTS_ROOT = Path("experiments/results")

# Output folder for the consolidated summary
SUMMARY_DIR = RESULTS_ROOT / "_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Final summary output files
CSV_PATH = SUMMARY_DIR / "model_comparison.csv"
MD_PATH = SUMMARY_DIR / "model_comparison.md"


# ---------- Helper functions ----------

def fmt(value: Optional[float], ndigits: int = 4) -> str:
    """
    Safely format numeric values for Markdown output.

    Returns 'N/A' when the value is missing or invalid.
    """
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return "N/A"


def safe_float(x) -> Optional[float]:
    """
    Convert a value to float safely.

    Returns None if conversion is not possible.
    """
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def detect_dataset(model_name: str, report_text: str) -> str:
    """
    Infer dataset name from model folder name and/or report content.

    This makes the summary more readable even when folder naming
    is not perfectly standardized.
    """
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

    # Fallback: guess from folder name only
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
    Detect whether a model is Visual, Audio, or AV fusion.

    Order matters because some folder names may contain overlapping terms.
    """
    # The combined text is conceptually available here if later needed
    (model_name + " " + report_text).lower()
    mn = model_name.lower()

    # Fusion models are checked first because they may include visual/audio keywords too
    if "fusion" in mn or "multimodal" in mn:
        return "AV fusion"

    # Common visual model keywords
    visual_backbones = ["xception", "mobilenet", "vit", "vision", "temporal"]
    if any(v in mn for v in visual_backbones):
        return "Visual"

    # Common audio model keywords
    audio_keywords = ["mfcc", "wav", "audio", "mel", "spectrogram"]
    if any(a in mn for a in audio_keywords):
        return "Audio"

    return "Unknown"


def detect_backbone(model_name: str, report_text: str) -> str:
    """
    Infer model backbone from folder name and report text.
    """
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
    """
    Search one experiment folder for likely report and metrics files.

    The function first checks common expected filenames, then falls back
    to a recursive search to stay robust across slightly different outputs.
    """
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

    # Fallback search for any report-like TXT file
    if report_path is None:
        txts = sorted(folder.glob("**/*report*.txt"))
        report_path = txts[0] if txts else None

    # Fallback search for any metrics-like JSON file
    if json_path is None:
        js = sorted(folder.glob("**/*metrics*.json"))
        json_path = js[0] if js else None

    return report_path, json_path


def parse_from_json(json_path: Path) -> Dict[str, Optional[float]]:
    """
    Extract key metrics from a JSON metrics file.

    Several alternative key names are supported so the script can
    handle outputs from different evaluation scripts.
    """
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
    """
    Extract key metrics from a plain-text evaluation report using regex.

    This acts as a fallback when JSON is missing or incomplete.
    """
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

    # Balanced accuracy may appear in slightly different naming styles
    m_bal = re.search(r"Balanced\s*Acc(?:uracy)?:\s*([0-9.]+)", report_text, re.IGNORECASE)
    out["balanced_accuracy"] = safe_float(m_bal.group(1)) if m_bal else None

    # MCC is useful for reliability analysis, especially with imbalance
    m_mcc = re.search(r"\bMCC\b\s*:\s*([0-9.\-]+)", report_text, re.IGNORECASE)
    out["mcc"] = safe_float(m_mcc.group(1)) if m_mcc else None

    return out


def merge_metrics(primary: Dict[str, Optional[float]], fallback: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """
    Merge two metric dictionaries.

    Values from the primary source are kept unless missing,
    in which case fallback values are used.
    """
    merged = dict(primary)
    for k, v in fallback.items():
        if merged.get(k) is None and v is not None:
            merged[k] = v
    return merged


def discover_models():
    """
    Discover experiment result folders under experiments/results.

    Certain non-model folders can be skipped explicitly.
    """
    models = []
    for folder in sorted(RESULTS_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name == "_summary":
            continue
        if folder.name.lower() in {"preprocessing_checks"}:
            continue
        models.append(folder)
    return models


def write_csv(rows):
    """
    Write the consolidated comparison table to CSV.

    This file is useful for spreadsheet-style analysis and reporting.
    """
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


def sort_rows_by_metric(rows, metric_index: int, desc: bool = True):
    """
    Sort rows by a given metric column.

    Rows with missing values are always pushed to the end so
    rankings remain meaningful.
    """
    rows_with = [r for r in rows if r[metric_index] is not None]
    rows_none = [r for r in rows if r[metric_index] is None]
    rows_with_sorted = sorted(rows_with, key=lambda r: r[metric_index], reverse=desc)
    return rows_with_sorted + rows_none


def best_row(rows, metric_index: int):
    """
    Return the best row according to a chosen metric.

    Missing values are ignored. Returns None if no valid rows exist.
    """
    candidates = [r for r in rows if r[metric_index] is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r[metric_index])


def group_by_dataset(rows):
    """
    Group result rows by dataset.

    Row schema:
      0 model
      1 dataset
      2 model_type
      3 backbone
      4 accuracy
      5 balanced_accuracy
      6 f1_weighted
      7 f1_macro
      8 auc
      9 ap
      10 mcc
      11 report_path
      12 JSON_path
    """
    groups = {}
    for r in rows:
        ds = r[1]
        groups.setdefault(ds, []).append(r)
    return groups


def is_visual(row) -> bool:
    """
    Check whether a row corresponds to a visual model.
    """
    return (row[2] or "").strip().lower() == "visual"


def short_row_ref(row) -> str:
    """
    Build a compact one-line summary string for a model row.

    This is used in executive summaries and rankings.
    """
    model, dataset, model_type, backbone = row[0], row[1], row[2], row[3]
    acc, bal, mcc = row[4], row[5], row[10]
    return f"**{model}** ({dataset}) — Type: {model_type}, Backbone: {backbone}, Acc={fmt(acc)}, BalAcc={fmt(bal)}, MCC={fmt(mcc)}"


def reliability_flags(row) -> list[str]:
    """
    Detect simple warning signs in the results.

    Examples:
    - very high accuracy but low balanced accuracy
    - MCC close to zero, suggesting weak real signal
    """
    flags = []
    acc = row[4]
    bal = row[5]
    mcc = row[10]

    if acc is not None and bal is not None:
        if acc >= 0.80 and bal <= 0.55:
            flags.append("High Acc but low BalAcc (possible class imbalance bias)")
    if mcc is not None:
        if abs(mcc) < 0.05:
            flags.append("MCC≈0 (weak correlation / unreliable)")
    return flags


def write_markdown(rows):
    """
    Write a human-readable Markdown summary.

    The report includes:
    - executive summary
    - best model per dataset
    - best visual model per dataset
    - global rankings
    - full comparison table
    """
    # ---- Sort once for global ranking ----
    rows_by_acc = sort_rows_by_metric(rows, metric_index=4, desc=True)
    best_overall = best_row(rows, metric_index=4)

    # ---- Dataset group summaries ----
    ds_groups = group_by_dataset(rows)

    best_per_ds = {}
    best_visual_per_ds = {}

    for ds, ds_rows in ds_groups.items():
        best_per_ds[ds] = best_row(ds_rows, metric_index=4)

        visual_rows = [r for r in ds_rows if is_visual(r)]
        best_visual_per_ds[ds] = best_row(visual_rows, metric_index=4) if visual_rows else None

    # Additional rankings useful for real-world reliability
    rows_by_balacc = sort_rows_by_metric(rows, metric_index=5, desc=True)
    rows_by_mcc = sort_rows_by_metric(rows, metric_index=10, desc=True)

    lines = [
        "# Model Comparison Summary\n",
        "Auto-generated from `experiments/results/*`.\n",
        "## Executive summary\n"
    ]

    # ---- Executive summary ----
    if best_overall:
        lines.append(f"- **Best overall (by Accuracy):** {short_row_ref(best_overall)}")
        flags = reliability_flags(best_overall)
        if flags:
            for fl in flags:
                lines.append(f"  - {fl}")
    else:
        lines.append("- **Best overall (by Accuracy):** N/A")

    lines.append("\n### Best model per dataset (by Accuracy)\n")
    for ds in sorted(best_per_ds.keys()):
        r = best_per_ds[ds]
        if r:
            lines.append(f"- **{ds}:** {short_row_ref(r)}")
            flags = reliability_flags(r)
            if flags:
                for fl in flags:
                    lines.append(f"  - {fl}")
        else:
            lines.append(f"- **{ds}:** N/A")

    lines.append("\n### Best visual model per dataset (by Accuracy)\n")
    for ds in sorted(best_visual_per_ds.keys()):
        r = best_visual_per_ds[ds]
        if r:
            lines.append(f"- **{ds}:** {short_row_ref(r)}")
            flags = reliability_flags(r)
            if flags:
                for fl in flags:
                    lines.append(f"  - {fl}")
        else:
            lines.append(f"- **{ds}:** N/A (no visual models detected)")

    # ---- Global rankings ----
    lines.append("\n## Global ranking\n")
    lines.append("### Ranking by Accuracy\n")
    for i, r in enumerate(rows_by_acc, start=1):
        lines.append(f"{i}. {short_row_ref(r)}")
        flags = reliability_flags(r)
        if flags:
            for fl in flags:
                lines.append(f"   - {fl}")

    # ---- Real-world deployment rankings ----
    lines.append("\n### Ranking by Balanced Accuracy (if available)\n")
    any_balacc = any(r[5] is not None for r in rows)
    if not any_balacc:
        lines.append("- No Balanced Accuracy values found in reports/metrics JSON.")
    else:
        shown = 0
        for i, r in enumerate(rows_by_balacc, start=1):
            if r[5] is None:
                continue
            lines.append(f"{i}. {short_row_ref(r)}")
            shown += 1
        if shown == 0:
            lines.append("- No Balanced Accuracy values found in usable rows.")

    lines.append("\n### Ranking by MCC (if available)\n")
    any_mcc = any(r[10] is not None for r in rows)
    if not any_mcc:
        lines.append("- No MCC values found in reports/metrics JSON.")
    else:
        shown = 0
        for i, r in enumerate(rows_by_mcc, start=1):
            if r[10] is None:
                continue
            lines.append(f"{i}. {short_row_ref(r)}")
            shown += 1
        if shown == 0:
            lines.append("- No MCC values found in usable rows.")

    # ---- Full table ----
    lines.append("\n## Full comparison table\n")
    lines.append("| Model | Dataset | Type | Backbone | Acc | Bal Acc | F1(w) | F1(m) | AUC | AP | MCC |")
    lines.append("|------|---------|------|----------|-----|---------|-------|-------|-----|----|-----|")

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
    """
    Main consolidation pipeline.

    This function:
    - discovers experiment result folders
    - finds report and metrics files
    - extracts and merges key metrics
    - infers dataset/model metadata
    - writes final CSV and Markdown summaries
    """
    model_folders = discover_models()
    if not model_folders:
        print("No model folders found under experiments/results/")
        return

    rows = []

    for folder in model_folders:
        model_name = folder.name
        report_path, json_path = find_report_and_metrics_json(folder)

        # Skip folders with no usable evaluation outputs
        if report_path is None and json_path is None:
            continue

        report_text = ""
        if report_path and report_path.exists():
            report_text = report_path.read_text(encoding="utf-8", errors="ignore")

        # JSON is treated as primary metric source when available
        metrics_from_json = {}
        if json_path and json_path.exists():
            try:
                metrics_from_json = parse_from_json(json_path)
            except Exception:
                metrics_from_json = {}

        # Text report acts as fallback
        metrics_from_text = parse_from_text(report_text)

        # Merge both sources so missing values can be filled where possible
        metrics = merge_metrics(metrics_from_json, metrics_from_text)

        # Infer dataset/model metadata for comparison table
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