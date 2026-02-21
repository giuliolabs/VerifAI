from __future__ import annotations

import argparse
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to curve CSV")
    parser.add_argument("--outdir", default=None, help="Output directory for plots (default: same folder as CSV)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(csv_path)
    if not rows:
        raise SystemExit("No data found in CSV.")

    # Epoch parsing with fallback index if missing
    epochs = []
    for i, r in enumerate(rows):
        e = r.get("epoch", "")
        if isinstance(e, str) and e.strip().isdigit():
            epochs.append(int(e))
        else:
            epochs.append(i + 1)

    def plot_if_present(keys: list[str], ylabel: str, title_suffix: str, out_suffix: str) -> None:
        present = any(k in rows[0] for k in keys)
        if not present:
            return

        plt.figure()
        for k in keys:
            if k in rows[0]:
                ys = [to_float(r.get(k, "")) for r in rows]
                # avoid totally empty plots
                if all(math.isnan(v) for v in ys):
                    continue
                plt.plot(epochs, ys, label=k)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title_suffix}: {csv_path.stem}")
        plt.legend()
        out = outdir / f"{csv_path.stem}_{out_suffix}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print("[OK] Saved:", out)

    plot_if_present(["train_loss", "val_loss"], "Loss", "Loss Curves", "loss")
    plot_if_present(["val_auc"], "AUC", "Validation AUC", "val_auc")
    plot_if_present(["val_f1"], "F1", "Validation F1", "val_f1")
    plot_if_present(["acc", "train_acc", "val_acc"], "Accuracy", "Accuracy", "accuracy")

if __name__ == "__main__":
    main()
