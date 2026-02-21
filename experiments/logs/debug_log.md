# VerifAI Debug / Error Tracking Log

This log documents bugs, investigation steps, and fixes across VerifAI:
data ingestion, preprocessing, training, evaluation, API, explainability, and user study.
Keeping this log improves reproducibility and demonstrates rigor for examiners.

---

## Common issues seen in this project (examples)

- Missing/invalid file paths (dataset folder names differ, OneDrive sync issues)
- CSV split mismatch (empty test CSV, wrong root paths)
- Torch dtype mismatch (Double vs Float) during eval
- Checkpoint compatibility (missing/unexpected keys; timm xception → legacy_xception warning)
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
Paste the key traceback line(s) or describe unexpected behavior.

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

