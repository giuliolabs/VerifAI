"""
Celeb-DF-v2 Test CSV Builder (Week 19)
=====================================

This script generates a CSV manifest for evaluating the Celeb-DF-v2
dataset within the VerifAI framework.

The manifest is built using the official dataset file:

    List_of_testing_videos.txt

This text file contains the official evaluation video list but often
includes numeric group prefixes that do not appear in the extracted
dataset folders.

Example entries in the official file:

    1 YouTube-real/00170.mp4
    2 Celeb-real/id0_0000.mp4
    3 Celeb-synthesis/id0_id16_0000.mp4

However, the actual dataset folder structure typically looks like:

    YouTube-real/
    Celeb-real/
    Celeb-synthesis/

This script removes the numeric prefixes so that paths correctly match
the real folders on disk.

------------------------------------------------
LABEL MAPPING
------------------------------------------------

Folder name → Label

    Celeb-synthesis  → fake
    Celeb-real       → real
    YouTube-real     → real

------------------------------------------------
OUTPUT
------------------------------------------------

The generated CSV file is saved to:

    data/splits/celebdfv2_test.csv

Columns:

    video_path,label,video_id

Example row:

    data/raw/Celeb-DF-v2/Celeb-synthesis/id0_id16_0000.mp4,fake,id0_id16_0000

------------------------------------------------
DEPENDENCIES
------------------------------------------------

Only Python standard library modules are required.

------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------

This script standardizes dataset manifests so that evaluation pipelines
can operate consistently across datasets such as:

    - FaceForensics++
    - Celeb-DF-v2
    - DeeperForensics
    - FakeAVCeleb

It also performs a simple path existence check to confirm that the
generated manifest matches the extracted dataset structure.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# -------------------------------------------------
# Standard Library Imports
# -------------------------------------------------

import os
import csv
import re


# -------------------------------------------------
# Dataset configuration
# -------------------------------------------------

# Root directory of the Celeb-DF-v2 dataset
DATASET_ROOT = os.path.join("data", "raw", "Celeb-DF-v2")

# Official test list included with the dataset
TEST_LIST = os.path.join(DATASET_ROOT, "List_of_testing_videos.txt")

# Output CSV path used by evaluation scripts
OUT_CSV = os.path.join("data", "splits", "celebdfv2_test.csv")


# -------------------------------------------------
# Utility Functions
# -------------------------------------------------

def normalize_rel_path(rel_path: str) -> str:
    """
    Normalize relative paths from the official test list.

    The official list often contains numeric prefixes such as:

        "1 YouTube-real/00170.mp4"

    This function removes the prefix and normalizes slashes so
    the path matches the real dataset folders.

    Example:

        "1 YouTube-real/00170.mp4"
            → "YouTube-real/00170.mp4"
    """

    text = rel_path.strip().replace("\\", "/")

    # Remove numeric prefix pattern "<digits><space>"
    text = re.sub(r"^\d+\s+", "", text)

    return text


def infer_label_from_path(rel_path: str) -> str:
    """
    Determine whether a video is real or fake based on its folder.

    Args:
        rel_path: normalized dataset-relative path

    Returns:
        "real", "fake", or "unknown"
    """

    rel_lower = rel_path.lower().replace("\\", "/")

    if rel_lower.startswith("celeb-synthesis/"):
        return "fake"

    if rel_lower.startswith("celeb-real/") or rel_lower.startswith("youtube-real/"):
        return "real"

    return "unknown"


# -------------------------------------------------
# Main CSV generation logic
# -------------------------------------------------

def main():
    """
    Build the Celeb-DF-v2 test CSV manifest.

    Steps:
    1. Read official test list
    2. Normalize paths
    3. Infer labels
    4. Save CSV manifest
    """

    if not os.path.exists(TEST_LIST):
        raise FileNotFoundError(f"Missing test list file: {TEST_LIST}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []

    # -------------------------------------------------
    # Read official test list
    # -------------------------------------------------

    with open(TEST_LIST, "r", encoding="utf-8") as f:

        for line in f:

            raw_rel_path = line.strip()

            if not raw_rel_path:
                continue

            # Normalize dataset-relative path
            rel_path = normalize_rel_path(raw_rel_path)

            # Construct absolute dataset path
            full_path = os.path.join(DATASET_ROOT, rel_path)

            # Infer real/fake label
            label = infer_label_from_path(rel_path)

            # Use filename stem as video_id
            video_id = os.path.splitext(os.path.basename(rel_path))[0]

            rows.append((full_path, label, video_id))

    # -------------------------------------------------
    # Write CSV manifest
    # -------------------------------------------------

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["video_path", "label", "video_id"])

        for full_path, label, video_id in rows:
            writer.writerow([full_path, label, video_id])

    # -------------------------------------------------
    # Sanity check
    # -------------------------------------------------

    exists_count = sum(
        1 for full_path, _, _ in rows if os.path.exists(full_path)
    )

    print(f"Wrote: {OUT_CSV} ({len(rows)} rows)")

    print(f"Paths that exist on disk: {exists_count}/{len(rows)}")

    # Print a few example rows for verification
    for row in rows[:3]:
        print("Sample:", row)


# -------------------------------------------------
# Script entry point
# -------------------------------------------------

if __name__ == "__main__":
    main()