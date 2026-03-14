"""
Internal API Test Runner – VerifAI (Week 18)
===========================================

This script performs a repeatable sanity test against the FastAPI backend
prediction endpoint (POST /api/predict). It uploads a single video file
to the API and prints the HTTP status code and JSON response.

The goal is to provide clear internal testing evidence for Week 18 of the
VerifAI project, demonstrating that the backend API is functioning
correctly and that requests are handled reliably.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
Required Python packages (install via pip):

    pip install requests

Tested with:
- Python 3.12+
- FastAPI backend
- Uvicorn server running locally

------------------------------------------------
INPUT
------------------------------------------------
Command-line arguments:

    --video
        Path to a local .mp4 file to upload.

    --url
        API endpoint URL.
        Default: http://127.0.0.1:8000/api/predict

    --timeout
        Request timeout in seconds.
        Default: 300 seconds.

------------------------------------------------
OUTPUT
------------------------------------------------
Printed to terminal:

    - HTTP status code
    - Raw response body (usually JSON)

This output can be copied into:

    experiments/week18/internal_testing_notes.md

------------------------------------------------
HOW TO RUN
------------------------------------------------
1) Start the backend server (from project root):

    python -m uvicorn backend.main:app --reload

2) Run this test script (from project root):

    python scripts/test_api_predict.py \
        --video "data/raw/.../example.mp4"

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- This script intentionally keeps testing simple and reproducible.
- It validates the API contract and server stability.
- Input file existence is checked before sending the request.
- The same video input should produce a consistent response format.
- Used alongside additional edge-case tests recorded in Week 18 notes.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Week: 18
Copyright © 2026 Giulio Labs
"""

# -------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------

# argparse is used for parsing command-line arguments
import argparse

# os is used for file system operations such as checking
# whether the input video file exists
import os

# -------------------------------------------------------
# Third-Party Imports
# -------------------------------------------------------

# requests is used to send HTTP POST requests to the API
import requests


# -------------------------------------------------------
# Main execution function
# -------------------------------------------------------
def main():
    """
    Entry point for the API testing script.

    This function:
    1. Parses command-line arguments
    2. Validates the input video file
    3. Sends a POST request to the FastAPI endpoint
    4. Prints the response status and body
    """

    # ---------------------------------------------
    # Parse command-line arguments
    # ---------------------------------------------
    parser = argparse.ArgumentParser(
        description="Internal VerifAI API test runner."
    )

    # Required argument: path to video file
    parser.add_argument(
        "--video",
        required=True,
        help="Path to a local .mp4 file to upload."
    )

    # Optional argument: API endpoint URL
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/api/predict",
        help="Prediction API endpoint."
    )

    # Optional argument: request timeout
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds."
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Validate input file
    # ---------------------------------------------
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"File not found: {args.video}")

    # ---------------------------------------------
    # Send POST request to API
    # ---------------------------------------------
    # Open the video file in binary mode
    with open(args.video, "rb") as f:

        # Prepare multipart form upload
        files = {
            "file": (
                os.path.basename(args.video),  # filename
                f,                             # file object
                "video/mp4"                    # MIME type
            )
        }

        # Send HTTP POST request
        response = requests.post(
            args.url,
            files=files,
            timeout=args.timeout
        )

    # ---------------------------------------------
    # Print API response
    # ---------------------------------------------
    print("Status:", response.status_code)
    print("Response:", response.text)


# -------------------------------------------------------
# Script entrypoint
# -------------------------------------------------------
if __name__ == "__main__":
    main()