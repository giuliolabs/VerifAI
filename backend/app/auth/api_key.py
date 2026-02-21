import os
from fastapi import Header, HTTPException

API_KEY = os.getenv("VERIFAI_API_KEY")  # Render

def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY is None:
        return  # auth disabled if not set
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
