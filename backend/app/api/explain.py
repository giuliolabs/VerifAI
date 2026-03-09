"""
Explainability Endpoint (Stub)
==============================

This endpoint is included as a placeholder to show that the system
has been designed with explainability in mind. However, the full
implementation is not required for this submission, so it currently
returns a simple status message.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Import APIRouter to create modular API routes
from fastapi import APIRouter

# Create a router instance to group related endpoints. This allows better organization of the API structure
router = APIRouter()


# Define a GET endpoint called /explain. This would normally return model explanation details
@router.get("/explain")
def explain_stub():
    # Since explainability is not implemented, return a structured placeholder response instead.
    return {
        "status": "not_implemented",  # Indicates the feature is scaffolded only
        "message": "Explainability is scaffolded but not required for this submission."
    }