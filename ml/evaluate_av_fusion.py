"""
Evaluate Multimodal Fusion Model (Audio + Video) on FakeAVCeleb_v1.2 (TEST split)
==================================================================================

Outputs (in experiments/results/fakeavceleb_av_fusion_v1):
- test_report.txt
- test_metrics.json
- test_per_category.csv

This version reports:
- overall binary metrics
- per-category FakeAVCeleb breakdown

Run (from project root):
    python -m ml.evaluate_av_fusion

Dependencies:
    pip install torch torchvision numpy pandas scikit-learn tqdm

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.av_data_loader import FakeAVCelebAVDataset
from ml.models.fusion.multimodal_fusion import MultimodalFusionModel
from ml.metrics import compute_binary_metrics, save_metrics_report


def _to_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.size(1) == 1:
        return logits.squeeze(1)
    return logits


def _torch_load_compat(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _safe_auc(y_true, y_score):
    try:
        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_ap(y_true, y_score):
    try:
        if len(set(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def _compute_category_metrics(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    rows = []

    for category in sorted(df["av_category"].dropna().unique().tolist()):
        sub = df[df["av_category"] == category].copy()

        y_true = sub["y_true"].astype(int).to_numpy()
        y_score = sub["y_score"].astype(float).to_numpy()
        y_pred = (y_score >= threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
        auc = _safe_auc(y_true, y_score)
        ap = _safe_ap(y_true, y_score)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "av_category": category,
            "num_samples": int(len(sub)),
            "num_real": int((sub["y_true"] == 0).sum()),
            "num_fake": int((sub["y_true"] == 1).sum()),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc) if not np.isnan(bal_acc) else float("nan"),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc),
            "average_precision": float(ap),
            "mcc": float(mcc) if not np.isnan(mcc) else float("nan"),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "mean_prob_fake": float(sub["y_score"].mean()),
        })

    return pd.DataFrame(rows)


@torch.no_grad()
def main():
    ckpt_path = Path("experiments/results/fakeavceleb_av_fusion_v1/best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt = _torch_load_compat(ckpt_path, device)
    mfcc_max_len = int(ckpt.get("mfcc_max_len", 200))

    # ---- data ----
    test_ds = FakeAVCelebAVDataset("test", strict=True, mfcc_max_len=mfcc_max_len)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # Need metadata for categories
    split_csv = Path("data/splits/fakeavceleb_test.csv")
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    split_df = pd.read_csv(split_csv)
    if "video_id" not in split_df.columns or "av_category" not in split_df.columns:
        raise ValueError("fakeavceleb_test.csv must contain at least 'video_id' and 'av_category' columns")

    video_to_category = dict(zip(split_df["video_id"].astype(str), split_df["av_category"].astype(str)))

    # ---- model ----
    model = MultimodalFusionModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    records = []

    for video, mfcc, y, video_ids in tqdm(test_loader, desc="Evaluating"):
        video = video.to(device)
        mfcc = mfcc.to(device)

        logits = _to_1d_logits(model(video, mfcc))
        probs = torch.sigmoid(logits).cpu().numpy()

        if isinstance(y, torch.Tensor):
            y_true_batch = y.cpu().numpy().astype(int).tolist()
        else:
            y_true_batch = [int(v) for v in y]

        for video_id, y_true, y_score in zip(video_ids, y_true_batch, probs.tolist()):
            category = video_to_category.get(str(video_id), "UNKNOWN")
            records.append({
                "video_id": str(video_id),
                "av_category": category,
                "y_true": int(y_true),
                "y_score": float(y_score),
            })

    eval_df = pd.DataFrame(records)
    y_true = eval_df["y_true"].astype(int).tolist()
    y_score = eval_df["y_score"].astype(float).tolist()

    # ---- overall metrics ----
    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        include_curves=True,
    )

    y_pred = (eval_df["y_score"].to_numpy() >= 0.5).astype(int)
    overall_bal_acc = balanced_accuracy_score(eval_df["y_true"], y_pred)
    overall_mcc = matthews_corrcoef(eval_df["y_true"], y_pred)

    print("\nAUDIO+VIDEO FUSION RESULTS (FakeAVCeleb AV Fusion v1)")
    print("Checkpoint:", ckpt_path)
    print("MFCC max len (batching):", mfcc_max_len)
    print("Num samples:", metrics.num_samples)
    print("Accuracy:", metrics.accuracy)
    print("Balanced accuracy:", overall_bal_acc)
    print("ROC-AUC:", metrics.auc_roc)
    print("Average precision:", metrics.ap)
    print("MCC:", overall_mcc)
    print("Confusion matrix:\n", np.array(metrics.confusion_matrix))
    print("\nReport:\n", metrics.report)

    txt_path, json_path = save_metrics_report(
        out_dir=out_dir,
        name="test",
        metrics=metrics,
        extra={
            "dataset": "FakeAVCeleb",
            "model": "AV Fusion v1",
            "mfcc_max_len": mfcc_max_len,
            "checkpoint": str(ckpt_path),
            "balanced_accuracy": overall_bal_acc,
            "mcc": overall_mcc,
        },
    )

    # ---- per-category metrics ----
    per_cat_df = _compute_category_metrics(eval_df, threshold=0.5)
    per_cat_path = out_dir / "test_per_category.csv"
    per_cat_df.to_csv(per_cat_path, index=False)

    print("\nPER-CATEGORY RESULTS")
    if len(per_cat_df) == 0:
        print("No per-category rows found.")
    else:
        for _, row in per_cat_df.iterrows():
            print(
                f"{row['av_category']}: "
                f"n={int(row['num_samples'])}, "
                f"acc={row['accuracy']:.4f}, "
                f"bal_acc={row['balanced_accuracy'] if pd.notna(row['balanced_accuracy']) else 'nan'}, "
                f"precision={row['precision']:.4f}, "
                f"recall={row['recall']:.4f}, "
                f"f1={row['f1']:.4f}, "
                f"mean_prob_fake={row['mean_prob_fake']:.4f}, "
                f"TN={int(row['tn'])}, FP={int(row['fp'])}, FN={int(row['fn'])}, TP={int(row['tp'])}"
            )

    print("\nSaved report ->", txt_path)
    print("Saved metrics json ->", json_path)
    print("Saved per-category csv ->", per_cat_path)


if __name__ == "__main__":
    main()
