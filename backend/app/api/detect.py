"""
Detection API Endpoint
======================

Receives uploaded video files and returns deepfake detection results.

Inference strategy:
- hybrid AV model when usable audio exists
- visual-only fallback when audio is unavailable

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# FastAPI tools used to create the route, accept uploaded files,
# return errors, and protect the endpoint with an API key
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

# Import the main inference function that performs the deepfake prediction
from backend.app.services.inference_service import run_inference

# Import the response schema so the returned JSON follows a fixed structure
from backend.app.models.schemas import PredictResponse

# Import API key protection so only authorized users can access this endpoint
from backend.app.auth.api_key import require_api_key

# Import logging function to store prediction activity in the database
from backend.app.db.events import log_prediction

# Create a router object to group related API endpoints together
router = APIRouter()

# Allowed video file formats for upload validation
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}

# Maximum accepted upload size in MB to avoid very large files being processed
MAX_UPLOAD_MB = 25


# Create a POST endpoint called /predict
# It returns data matching PredictResponse
# The dependency ensures that a valid API key is required before access is granted
@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict(file: UploadFile = File(...)):
    """
    Run deepfake detection on an uploaded video.
    """

    # Check that the uploaded file actually has a filename
    # If not, it usually means no file was provided in the request
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Convert filename to lowercase so extension checking is case-insensitive
    filename_lower = file.filename.lower()

    # Make sure the uploaded file ends with one of the supported extensions
    # This is a simple first validation before reading the file content
    if not any(filename_lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Read the first 512 bytes of the file
    # This is enough for a quick content check without loading the whole file yet
    first_chunk = await file.read(512)

    # If nothing is read, the uploaded file is empty
    if not first_chunk:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Extra validation for MP4 files:
    # many valid MP4 containers contain the 'ftyp' signature near the beginning
    # This helps detect files that only pretend to be .mp4 by name
    if filename_lower.endswith(".mp4") and b"ftyp" not in first_chunk:
        await file.seek(0)
        raise HTTPException(
            status_code=400,
            detail="Invalid MP4 file. The uploaded file does not look like a valid MP4 container.",
        )

    # Convert the upload limit from MB into bytes for comparison
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    # Start counting the file size using the bytes already read
    size_so_far = len(first_chunk)

    # Continue reading the file in 1 MB chunks until:
    # - the whole file is read, or
    # - the size goes over the allowed maximum
    # This avoids storing the entire file in memory at once
    while size_so_far <= max_bytes:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size_so_far += len(chunk)

    # Reset the file pointer back to the start
    # This is important because the file will need to be read again during inference
    await file.seek(0)

    # If the file size is larger than the allowed limit,
    # return HTTP 413 (Payload Too Large)
    if size_so_far > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB.",
        )

    try:
        # Run the deepfake detection process on the uploaded file
        result = run_inference(file)

    # If the inference service already raised a proper HTTPException,
    # re-raise it so the original error response is preserved
    except HTTPException:
        raise

    # Catch any unexpected internal errors and return a generic server error
    # Including the exception type can help with debugging
    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exception).__name__}",
        ) from exception

    # The model returns probability as a percentage string, e.g. "87.5%"
    # For database logging, convert it into a numeric value between 0 and 1
    numeric_prob = float(result["prob_fake"].replace("%", "")) / 100.0

    # Save the prediction result for monitoring, analysis, or audit purposes
    log_prediction(file.filename, result["label"], numeric_prob)

    # Return only the main fields expected by the API response schema
    return {
        "label": result["label"],
        "prob_fake": result["prob_fake"],
        "mode": result["mode"],
    }
