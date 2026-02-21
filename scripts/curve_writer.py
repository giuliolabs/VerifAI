from __future__ import annotations

from pathlib import Path
import csv

def append_curve_row(csv_path: str | Path, row: dict) -> None:
    """
    Appends a dict row to a CSV file (creates file + header if missing).

    VerifAI note:
    - Keeps a stable header across appends.
    - If new keys appear later, they are ignored (so plots stay consistent).
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    # Read existing header
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if not header:
        header = list(row.keys())

    # Only write fields that exist in header
    safe_row = {k: row.get(k, "") for k in header}

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(safe_row)
