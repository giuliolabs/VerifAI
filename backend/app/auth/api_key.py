"""
API Key Authentication Utility
==============================

This module provides simple API key protection for sensitive endpoints.
It checks whether incoming requests include a valid API key in the header.
If no API key is configured in the environment, authentication is disabled
to allow flexibility during development or testing.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# Import os to access environment variables
import os

# Import Header to read HTTP headers and HTTPException to return proper errors
from fastapi import Header, HTTPException


# Read the API key from environment variables (e.g. Render)
# This avoids hardcoding sensitive credentials directly in the source code
API_KEY = os.getenv("VERIFAI_API_KEY")  # Render


def require_api_key(x_api_key: str | None = Header(default=None)):
    """
    Dependency function that validates the provided API key.
    It is designed to be used with FastAPI's Depends() system.
    """

    # If no API key is configured in the environment, authentication is automatically disabled.
    # This is useful for local development where protection may not be required.
    if API_KEY is None:
        return  # auth disabled if not set

    # Compare the provided header value with the expected API key.
    # If they do not match, reject the request with HTTP 401 (Unauthorized).
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")