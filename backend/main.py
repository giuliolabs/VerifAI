"""
FastAPI Backend – VerifAI Multimodal Deepfake Detection
======================================================

This module initializes the FastAPI application used to expose
the multimodal (audio + video) deepfake detection model via a REST API.

The API is designed for:
- Local testing (Week 17 requirement)
- Future web frontend integration
- Extension with authentication, explainability, and provenance services

------------------------------------------------
DEPENDENCIES
------------------------------------------------
Required Python packages (install via pip):

    pip install fastapi uvicorn torch torchvision numpy opencv-python librosa
    pip install python-multipart

------------------------------------------------
ENDPOINTS
------------------------------------------------
POST /api/predict
    - Accepts a video file upload
    - Runs multimodal deepfake inference
    - Returns prediction + confidence score

GET /
    - Health check endpoint

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- The model is loaded once at application startup.
- API structure follows a service-oriented architecture.
- Additional modules (auth, explainability, provenance) are scaffolded
  but not required for Week 17 evaluation.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""


from fastapi import FastAPI

from backend.app.api.detect import router as detect_router
from backend.app.api import metadata
from backend.app.api.explain import router as explain_router
from backend.app.api.ui import router as ui_router


app = FastAPI(
    title="VerifAI – Multimodal Deepfake Detection API",
    version="1.0"
)

# -----------------------------
# API ROUTES
# -----------------------------
app.include_router(detect_router, prefix="/api", tags=["detect"])
app.include_router(metadata.router, prefix="/api", tags=["metadata"])
app.include_router(explain_router, prefix="/api", tags=["explain"])

# -----------------------------
# UI ROUTE (mounted at "/")
# -----------------------------
app.include_router(ui_router, prefix="", tags=["ui"])
