"""
FastAPI Backend – VerifAI Multimodal Deepfake Detection
======================================================

This module initializes the FastAPI application used to expose
the multimodal deepfake detection model via a REST API.

The API is designed for:
- Local testing (Week 17)
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
- Additional modules (auth, explainability, provenance) are scaffolded.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Import FastAPI to create the main web application instance
from fastapi import FastAPI

# Import StaticFiles to serve frontend assets such as logos or CSS
from fastapi.staticfiles import StaticFiles

# Import modular routers to maintain a clean service-oriented architecture
from backend.app.api.detect import router as detect_router
from backend.app.api import metadata
from backend.app.api.explain import router as explain_router
from backend.app.api.ui import router as ui_router

# Import database initialization for prediction logging
from backend.app.db.events import init_db


# Create the main FastAPI application instance.
# Metadata such as title and version improves automatic API documentation.
app = FastAPI(
    title="VerifAI – Multimodal Deepfake Detection API",
    version="1.0",
)


# -----------------------------
# Startup (initialize SQLite audit database)
# -----------------------------

# This function runs automatically when the application starts.
# It ensures the database is ready before any prediction requests are handled.
@app.on_event("startup")
def startup() -> None:
    init_db()
    print("VerifAI DB initialized")


# -----------------------------
# Static files (logo, UI assets, etc.)
# -----------------------------

# Mount a directory to serve static frontend resources.
# This allows the API to also support a simple web interface if needed.
app.mount("/static", StaticFiles(directory="backend/app/static"), name="static")


# -----------------------------
# API routes
# -----------------------------

# Include the prediction endpoint under the "/api" prefix.
# Tags help organize endpoints in the automatically generated Swagger UI.
app.include_router(detect_router, prefix="/api", tags=["detect"])

# Include metadata-related routes (e.g., health checks, system info).
app.include_router(metadata.router, prefix="/api", tags=["metadata"])

# Include explainability scaffold endpoint.
app.include_router(explain_router, prefix="/api", tags=["explain"])


# -----------------------------
# UI (mounted at root "/")
# -----------------------------

# This allows users to access a browser-based interface without needing a separate server.
app.include_router(ui_router, prefix="", tags=["ui"])