"""
Detection API Endpoint
======================

Defines the REST endpoint responsible for receiving video uploads
and returning deepfake detection results.

The endpoint delegates all inference to the inference service:
- Audio-Visual inference is used when usable audio is present
- Visual-only fallback is used when audio is absent

------------------------------------------------
DEPENDENCIES
------------------------------------------------
    pip install fastapi torch

------------------------------------------------
ENDPOINT
------------------------------------------------
POST /api/predict
    Input:
        - Multipart video file (mp4, avi, mov)
    Output:
        - JSON with prediction label and fake probability

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- This layer contains no ML logic.
- All inference is delegated to the inference service.
- Robust input validation prevents common 500/502 failures on hosted deployments.
- Includes max upload size guard (free-tier friendly).
- Includes MP4 signature validation (ftyp) to reject malformed uploads early.
- Designed to be thin, testable, and extensible.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from backend.app.services.inference_service import run_inference
from backend.app.models.schemas import PredictResponse
from backend.app.auth.api_key import require_api_key
from backend.app.db.events import log_prediction

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
MAX_UPLOAD_MB = 25


@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict(file: UploadFile = File(...)):
    """
    Run deepfake detection on an uploaded video.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename_lower = file.filename.lower()

    if not any(filename_lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    first_chunk = await file.read(512)
    if not first_chunk:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if filename_lower.endswith(".mp4") and b"ftyp" not in first_chunk:
        await file.seek(0)
        raise HTTPException(
            status_code=400,
            detail="Invalid MP4 file. The uploaded file does not look like a valid MP4 container.",
        )

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    size_so_far = len(first_chunk)

    while size_so_far <= max_bytes:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size_so_far += len(chunk)

    await file.seek(0)

    if size_so_far > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB.",
        )

    try:
        label, prob = run_inference(file)
    except HTTPException:
        raise
    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exception).__name__}",
        ) from exception

    log_prediction(file.filename, label, float(prob))

    return {"label": label, "prob_fake": float(prob)}
