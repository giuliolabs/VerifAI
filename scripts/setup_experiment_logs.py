from __future__ import annotations

from pathlib import Path
from datetime import datetime

# Repo root = parent of /scripts
ROOT = Path(__file__).resolve().parents[1]

# Central logs directory for the whole project (NOT just ablations)
LOGS_DIR = ROOT / "experiments" / "logs"

DEBUG_TEMPLATE = """# VerifAI Debug / Error Tracking Log

This log documents bugs, investigation steps, and fixes across VerifAI:
data ingestion, preprocessing, training, evaluation, API, explainability, and user study.
Keeping this log improves reproducibility and demonstrates rigour for examiners.

---

## Common issues seen in this project (examples)

- Missing/invalid file paths (dataset folder names differ, OneDrive sync issues)
- CSV split mismatch (empty test CSV, wrong root paths)
- Torch dtype mismatch (Double vs Float) during eval
- Checkpoint compatibility (missing/unexpected keys; timm xception -> legacy_xception warning)
- Large archive extraction issues (tar/zip join problems, memory limits)
- Frame/audio extraction failures for particular videos (corrupt or unsupported codec)

---

## Entry Template

### Date:
### Component:
(e.g., pipelines/splits, pipelines/video, ml/train, scripts/eval_cross_dataset, backend API, explainability)

### Context:
What were you trying to do? Which dataset/model/checkpoint/config?

### Symptom / Error:
Paste the key traceback line(s) or describe unexpected behaviour.

### Investigation:
Steps taken (commands run, files inspected, sanity checks).

### Root Cause:
What actually caused it.

### Fix:
What changed (file/function + short explanation).

### Verification:
How you confirmed it (rerun command, metrics, file counts, plots).

### Notes:
Any follow-ups / improvements.

---

## Entries

"""

def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Debug diary
    debug_log = LOGS_DIR / "debug_log.md"
    if not debug_log.exists():
        debug_log.write_text(DEBUG_TEMPLATE, encoding="utf-8")

    # Seed registry
    seeds_file = LOGS_DIR / "verifai_random_seeds.tsv"
    if not seeds_file.exists():
        seeds_file.write_text(
            "seed\tnote\ttimestamp\n",
            encoding="utf-8",
        )

    # Run logs + errors
    (LOGS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "errors").mkdir(parents=True, exist_ok=True)

    # Training curves
    (LOGS_DIR / "training_curves").mkdir(parents=True, exist_ok=True)

    # Readme
    readme = LOGS_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            "# VerifAI Experiment Logs\n\n"
            "Central location for reproducibility artifacts:\n\n"
            "- `debug_log.md`: structured debugging diary\n"
            "- `verifai_random_seeds.tsv`: seeds registry\n"
            "- `runs/`: JSON run metadata (configs, checkpoints, datasets, metrics)\n"
            "- `errors/`: JSON exception logs with tracebacks\n"
            "- `training_curves/`: per-run CSV curves + generated plots\n",
            encoding="utf-8",
        )

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[OK] VerifAI logs scaffold ready at: {LOGS_DIR} ({stamp})")

if __name__ == "__main__":
    main()
