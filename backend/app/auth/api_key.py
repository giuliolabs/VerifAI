import os
from fastapi import Header, HTTPException

API_KEY = os.getenv("verifai_2026_v1.0_key_2102")  # Render

def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY is None:
        return  # auth disabled if not set
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
