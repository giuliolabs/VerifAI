# VerifAI Debug / Error Tracking Log

This log documents bugs, investigation steps, and fixes across VerifAI:
data ingestion, preprocessing, training, evaluation, API, explainability, and user study.
Keeping this log improves reproducibility and demonstrates rigor for examiners.

---

## Common issues seen in this project

- Missing/invalid file paths (dataset folder names differ, OneDrive sync issues)
- CSV split mismatch (empty test CSV, wrong root paths)
- Torch dtype mismatch (Double vs Float) during eval
- Checkpoint compatibility (missing/unexpected keys; timm xception → legacy_xception warning)
- Large archive extraction issues (tar/zip join problems, memory limits)
- Frame/audio extraction failures for particular videos (corrupt or unsupported codec)

---

### Date: 21/02/2026
### Component: Backend API (Render deployment)

### Context: 
Deploy VerifAI FastAPI backend on Render and run /api/predict (multimodal inference: video frames + 
MFCC audio) using the checkpoint experiments/results/fakeavceleb_av_fusion_v1/best_model.pt. Tested via UI and 
PowerShell curl.exe uploads.

### Symptom / Error: 
1. UI returned HTTP 502 during upload/inference.
2. Render sent email: “instance exceeded its memory limit, triggered automatic restart.”
3. Logs showed routing errors during sleep/cold start (e.g., x-render-routing: dynamic-hibernate-error-503) and requests 
failing while the service restarted.

### Investigation:
1. Verified DNS routing (initially inconsistent due to local DNS resolver/router cache; fixed by switching Windows DNS 
to Google DNS).
2. Tested API directly with PowerShell curl.exe including X-API-Key.
3. Confirmed redirects between apex and www domains and fixed tests by targeting the canonical host.
4. Checked Render logs/alerts and correlated 502s with memory-limit restarts during inference.

### Root Cause:
Render free-tier resources (~512MB RAM, very low CPU) are insufficient for multimodal PyTorch inference + video decoding
 + audio MFCC extraction. Peak RAM spikes during preprocessing/inference cause the process to be killed (OOM), leading 
 to 502 and automatic restarts.

### Fix:
1. Operational mitigations: use small test clips, warm the service before requests, force single worker to avoid 
multiple model copies.
2. Code mitigations (planned/implemented): reduce peak memory (CPU-only execution, limit torch threads, 
torch.inference_mode(), explicit tensor cleanup).
3. Deployment fix (planned): upgrade instance (target 2GB RAM / 1 CPU) to run inference reliably.

### Verification:
1. Confirmed correct domain routing and API access via curl.exe (no Apache/Bluehost interference).
2. Verified API-key enforcement (401 without key, accepted with key).
3. After DNS fix, requests consistently reached Render/uvicorn; remaining failures aligned with OOM restarts during 
inference.

### Notes:
1. Free-tier hibernation can produce transient 503s; repeated requests after wake-up are expected.
2. For stable demo: upgrade resources or simplify preprocessing (fewer frames, smaller resize, optional audio 
disable) to keep inference within RAM limits.

---
