"""
Health Check Endpoint
=====================

This endpoint is used to verify that the API service is running correctly.
It provides a simple response that can be used for monitoring, testing,
or deployment checks (e.g. confirming the server is live).

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Import APIRouter to define modular API routes
from fastapi import APIRouter

# Create a router instance to organize endpoints separately. This keeps the API structure clean and scalable
router = APIRouter()


# Define a GET endpoint called /health
# This is used to check if the service is operational
@router.get("/health")
def health():
    # Return a simple JSON response indicating:
    # - the system status
    # - the name of the service
    # - the project development week (for version tracking in this coursework)
    return {
        "status": "ok",          # Confirms the API is running without issues
        "service": "VerifAI",    # Identifies the current backend service
        "week": 18               # Indicates the development milestone
    }