"""
API Response Schemas
====================

This module defines the structured response models used by the VerifAI API.
Pydantic models are used to enforce consistent data validation and ensure
that all responses follow a clearly defined format.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Import BaseModel to define structured data models
# Import Field to provide additional metadata such as examples
from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """
    Schema returned after a successful deepfake prediction.
    This ensures the API response always follows a fixed structure.
    """

    # Classification label returned by the model (e.g. real or fake)
    # Examples are included for automatic API documentation (Swagger UI)
    label: str = Field(..., examples=["real", "fake"])

    # Probability of the video being fake, formatted as a percentage string.
    # Stored as string because the API presents it in user-friendly format.
    prob_fake: str = Field(..., examples=["82.91%"])

    # Indicates which inference mode was used:
    # - hybrid_av: audio + visual model
    # - visual_only_fallback: visual-only model
    mode: str = Field(..., examples=["hybrid_av", "visual_only_fallback"])


class ErrorResponse(BaseModel):
    """
    Standard error response schema.
    Used to provide structured error messages when requests fail.
    """

    # Contains a human-readable explanation of the error
    detail: str