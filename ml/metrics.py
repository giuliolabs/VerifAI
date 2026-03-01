"""
Reusable metrics utilities for the Deepfake Detection Framework.

Why this file matters:
- Keeps evaluation consistent across all models (video/audio/fusion).
- Produces report-ready outputs (txt + JSON + optional plots).
- Helps you demonstrate a "framework" rather than one-off scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import json
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


@dataclass
class BinaryMetrics:
    num_samples: int
    accuracy: float
    auc_roc: float
    ap: float  # average precision (PR-AUC-ish)
    threshold: float
    confusion_matrix: List[List[int]]
    report: str

    # Optional curve data (useful for plots, ablations, and appendix)
    fpr: Optional[List[float]] = None
    tpr: Optional[List[float]] = None
    roc_thresholds: Optional[List[float]] = None
    precision: Optional[List[float]] = None
    recall: Optional[List[float]] = None
    pr_thresholds: Optional[List[float]] = None


def _to_float_list(arr: np.ndarray) -> List[float]:
    return [float(x) for x in arr.reshape(-1)]


def _safe_auc_roc(y_true: List[int], y_score: List[float]) -> float:
    # roc_auc_score breaks if y_true has only one class
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_binary_metrics(
    y_true: List[int],
    y_score: List[float],
    threshold: float = 0.5,
    include_curves: bool = True,
) -> BinaryMetrics:
    """
    Compute standard binary classification metrics from:
      - y_true: list of 0/1 labels
      - y_score: list of probabilities/scores in [0,1]

    threshold: decision threshold for y_pred
    include_curves: include ROC/PR curves for plotting/report appendix
    """
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must be the same length.")

    y_pred = [1 if float(p) >= threshold else 0 for p in y_score]

    acc = float(accuracy_score(y_true, y_pred))
    auc = _safe_auc_roc(y_true, y_score)

    # Average precision (works even when ROC-AUC is nan sometimes, but can still be nan if degenerate)
    try:
        ap = float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")
    except Exception:
        ap = float("nan")

    cm = confusion_matrix(y_true, y_pred).tolist()
    rep = classification_report(y_true, y_pred, digits=4)

    metrics = BinaryMetrics(
        num_samples=len(y_true),
        accuracy=acc,
        auc_roc=auc,
        ap=ap,
        threshold=float(threshold),
        confusion_matrix=cm,
        report=rep,
    )

    if include_curves and len(set(y_true)) > 1:
        fpr, tpr, roc_th = roc_curve(y_true, y_score)
        prec, rec, pr_th = precision_recall_curve(y_true, y_score)

        metrics.fpr = _to_float_list(fpr)
        metrics.tpr = _to_float_list(tpr)
        metrics.roc_thresholds = _to_float_list(roc_th)
        metrics.precision = _to_float_list(prec)
        metrics.recall = _to_float_list(rec)
        metrics.pr_thresholds = _to_float_list(pr_th)

    return metrics


def aggregate_video_scores_mean(
    frame_probs: Dict[str, List[float]],
    video_label: Dict[str, int],
) -> Tuple[List[int], List[float], List[str]]:
    """
    Convert frame-level probabilities to video-level scores
    by averaging probs per video.

    Returns:
      y_true, y_score, video_ids (sorted)
    """
    video_ids = sorted(frame_probs.keys())
    y_true: List[int] = []
    y_score: List[float] = []

    for vid in video_ids:
        y_true.append(int(video_label[vid]))
        y_score.append(float(np.mean(frame_probs[vid])))

    return y_true, y_score, video_ids


def save_metrics_report(
    out_dir: Path,
    name: str,
    metrics: BinaryMetrics,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """
    Saves:
      - <name>_report.txt (human-friendly)
      - <name>_metrics.json (machine-friendly)

    Returns: (txt_path, json_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{name}_report.txt"
    json_path = out_dir / f"{name}_metrics.json"

    # ---- TXT (report-ready) ----
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{name.upper()} RESULTS\n")
        f.write(f"Num samples: {metrics.num_samples}\n")
        f.write(f"Accuracy: {metrics.accuracy}\n")
        f.write(f"ROC-AUC: {metrics.auc_roc}\n")
        f.write(f"Average precision: {metrics.ap}\n")
        f.write(f"Threshold: {metrics.threshold}\n")
        f.write(f"Confusion matrix:\n{np.array(metrics.confusion_matrix)}\n\n")
        f.write(metrics.report)

        if extra:
            f.write("\n\nEXTRA\n")
            for key, value in extra.items():
                f.write(f"{key}: {value}\n")

    # ---- JSON (for plotting/ablation tables) ----
    payload = asdict(metrics)
    if extra:
        payload["extra"] = extra

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return txt_path, json_path
