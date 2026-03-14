"""
Training Curve CSV Logger – VerifAI
===================================

This utility function appends a training metric row to a CSV file while
maintaining a stable column structure.

It is used during training runs to record metrics such as:

    epoch, train_loss, val_loss, val_auc, val_f1, train_acc, val_acc

The function automatically:
- Creates the CSV file if it does not exist
- Writes the header on first creation
- Appends rows for subsequent epochs
- Ensures column order stays consistent across appends

------------------------------------------------
WHY THIS EXISTS
------------------------------------------------
Training scripts often log metrics after each epoch. This helper
ensures that logging is:

- consistent across experiments
- safe when writing incrementally
- compatible with plotting scripts later

It also avoids column corruption when new keys appear during logging.

------------------------------------------------
USAGE EXAMPLE
------------------------------------------------

from scripts.curve_logger import append_curve_row

append_curve_row(
    "experiments/logs/training_curves/xception_run1.csv",
    {
        "epoch": 1,
        "train_loss": 0.62,
        "val_loss": 0.55,
        "val_auc": 0.91
    }
)

------------------------------------------------
NOTES
------------------------------------------------
- The CSV header is determined by the first row written.
- If later rows contain new keys, they are ignored to preserve
  a consistent column structure.
- This keeps downstream plotting scripts stable.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# -------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------

# pathlib provides platform-independent file handling
from pathlib import Path

# csv module is used to write rows into CSV format
import csv


# -------------------------------------------------------
# Main logging function
# -------------------------------------------------------
def append_curve_row(csv_path: str | Path, row: dict) -> None:
    """
    Append a dictionary row to a CSV file.

    If the CSV file does not exist, it is created along with
    a header derived from the keys of the first row.

    If the CSV already exists, the function:
    - reads the existing header
    - ensures only known columns are written
    - appends the new row safely

    Args:
        csv_path:
            Path to the CSV file where metrics should be logged.

        row:
            Dictionary containing metric values for a training epoch.

    Example row structure:
        {
            "epoch": 5,
            "train_loss": 0.43,
            "val_loss": 0.38,
            "val_auc": 0.94
        }
    """

    # Convert path to a Path object
    path = Path(csv_path)

    # Ensure the directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Case 1: CSV does not exist -> create it
    # -------------------------------------------------------
    if not path.exists():

        with path.open("w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(f, fieldnames=list(row.keys()))

            # Write header row
            writer.writeheader()

            # Write first data row
            writer.writerow(row)

        return

    # -------------------------------------------------------
    # Case 2: CSV already exists -> append safely
    # -------------------------------------------------------

    # Read existing header to preserve column order
    with path.open("r", encoding="utf-8") as f:

        reader = csv.reader(f)

        header = next(reader, None)

    # If header somehow missing, regenerate from row keys
    if not header:
        header = list(row.keys())

    # -------------------------------------------------------
    # Create a "safe row"
    # -------------------------------------------------------
    # Only write values for existing header columns.
    # New keys that appear later are ignored to avoid
    # changing the CSV schema.
    safe_row = {k: row.get(k, "") for k in header}

    # -------------------------------------------------------
    # Append row to CSV
    # -------------------------------------------------------
    with path.open("a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=header)

        writer.writerow(safe_row)