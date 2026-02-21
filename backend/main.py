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
from fastapi.staticfiles import StaticFiles

from backend.app.api.detect import router as detect_router
from backend.app.api import metadata
from backend.app.api.explain import router as explain_router
from backend.app.api.ui import router as ui_router
from backend.app.db.events import init_db


app = FastAPI(
    title="VerifAI – Multimodal Deepfake Detection API",
    version="1.0",
)

# -----------------------------
# Startup (init SQLite audit DB)
# -----------------------------
@app.on_event("startup")
def startup() -> None:
    init_db()
    print("VerifAI DB initialized")


# -----------------------------
# Static files (logo, etc.)
# -----------------------------
app.mount("/static", StaticFiles(directory="backend/app/static"), name="static")


# -----------------------------
# API routes
# -----------------------------
app.include_router(detect_router, prefix="/api", tags=["detect"])
app.include_router(metadata.router, prefix="/api", tags=["metadata"])
app.include_router(explain_router, prefix="/api", tags=["explain"])


# -----------------------------
# UI (mounted at "/")
# -----------------------------
app.include_router(ui_router, prefix="", tags=["ui"])