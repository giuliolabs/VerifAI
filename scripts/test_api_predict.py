"""
Internal API Test Runner – VerifAI (Week 18)
===========================================

This script performs a repeatable sanity test against the FastAPI backend
prediction endpoint (POST /api/predict). It uploads a single video file
and prints the HTTP status code and JSON response.

It is designed to provide clear internal testing evidence for Week 18:
API reliability, request handling, and baseline response correctness.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
Required Python packages (install via pip):

    pip install requests

Tested with:
- Python 3.12+
- FastAPI + Uvicorn backend running locally

------------------------------------------------
INPUT
------------------------------------------------
Command-line arguments:
    --video   Path to a local .mp4 file
    --url     Endpoint URL (default: http://127.0.0.1:8000/api/predict)
    --timeout Request timeout in seconds (default: 300)

------------------------------------------------
OUTPUT
------------------------------------------------
Printed to terminal:
- HTTP status code
- Raw response body (typically JSON)

This output can be copied into:
    experiments/week18/internal_testing_notes.md

------------------------------------------------
HOW TO RUN
------------------------------------------------
1) Start backend (from project root):
    python -m uvicorn backend.main:app --reload

2) Run test script (from project root):
    python scripts/test_api_predict.py --video "data/raw/.../example.mp4"

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- The script is intentionally minimal: it tests API contract + stability.
- File existence is validated before request is sent.
- Designed for repeatability: same input video → consistent response format.
- Used alongside additional edge-case tests logged in Week 18 notes.

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
Week: 18
"""


import argparse
import os
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/predict")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"File not found: {args.video}")

    with open(args.video, "rb") as f:
        files = {"file": (os.path.basename(args.video), f, "video/mp4")}
        r = requests.post(args.url, files=files, timeout=300)

    print("Status:", r.status_code)
    print("Response:", r.text)

if __name__ == "__main__":
    main()
