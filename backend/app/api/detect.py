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
- Week 18 improvement: robust input validation to prevent 500 errors for invalid files.
- Additional Week 18 improvement: MP4 signature validation (ftyp) to reject malformed uploads early.
- Designed to be thin, testable, and extensible.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.inference_service import run_inference

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}  # keep aligned with your report claims


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Run multimodal deepfake detection on an uploaded video.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename_lower = file.filename.lower()

    # --- Week 18: file extension validation ---
    if not any(filename_lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # --- Week 18: empty/malformed upload validation ---
    # Read enough bytes to detect empty uploads and basic MP4 structure, then reset stream position.
    header = await file.read(512)
    await file.seek(0)

    if not header:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # If user claims it's an MP4, do a lightweight signature check:
    # MP4 containers typically contain a 'ftyp' box near the start of the file.
    if filename_lower.endswith(".mp4") and b"ftyp" not in header:
        raise HTTPException(
            status_code=400,
            detail="Invalid MP4 file. The uploaded file does not look like a valid video."
        )

    # --- Run inference ---
    try:
        label, prob = run_inference(file)
    except HTTPException:
        # If your inference_service raises HTTPException, preserve it
        raise
    except Exception as exception:
        # Convert unexpected inference errors into a clearer response (still 500)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exception).__name__}"
        ) from exception

    return {
        "label": label,
        "prob_fake": float(prob)
    }
