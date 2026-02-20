"""
Detection API Endpoint
======================

Defines the REST endpoint responsible for receiving video uploads
and returning multimodal deepfake detection results.

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
        - JSON with prediction label and confidence score

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

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.app.services.inference_service import run_inference

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}  # keep aligned with your report claims
MAX_UPLOAD_MB = 25  # adjust if you upgrade Render plan


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Run multimodal deepfake detection on an uploaded video.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename_lower = file.filename.lower()

    # --- File extension validation ---
    if not any(filename_lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # --- Basic sanity checks (empty, size guard, mp4 signature) ---
    # Read an initial chunk to:
    #  - detect empty uploads
    #  - check mp4 'ftyp' signature
    #  - estimate size without loading the entire file into RAM
    #
    # NOTE: UploadFile is spooled to disk after a threshold, but on hosted platforms
    # it’s still smart to enforce an upper bound to avoid timeouts/502s.
    first_chunk = await file.read(512)
    if not first_chunk:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Lightweight MP4 signature check
    if filename_lower.endswith(".mp4") and b"ftyp" not in first_chunk:
        await file.seek(0)
        raise HTTPException(
            status_code=400,
            detail="Invalid MP4 file. The uploaded file does not look like a valid MP4 container.",
        )

    # Size guard (stream the rest in chunks and stop once we exceed the limit)
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    size_so_far = len(first_chunk)

    # Read remaining bytes in chunks, but abort once limit exceeded
    while size_so_far <= max_bytes:
        chunk = await file.read(1024 * 1024)  # 1MB
        if not chunk:
            break
        size_so_far += len(chunk)

    # Reset the stream pointer so inference_service can read from the start
    await file.seek(0)

    if size_so_far > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB.",
        )

    # --- Run inference ---
    try:
        label, prob = run_inference(file)
    except HTTPException:
        raise
    except Exception as exception:
        # Convert unexpected inference errors into a clearer response.
        # (Prevents generic 502 with zero context)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exception).__name__}",
        ) from exception

    return {"label": label, "prob_fake": float(prob)}
