"""
Celeb-DF-v2 Test CSV Builder (Week 19)
=====================================

Builds a CSV manifest for Celeb-DF-v2 evaluation using the official
List_of_testing_videos.txt file included with the dataset.

IMPORTANT:
The official list often contains numbered group prefixes, e.g.
    "1 YouTube-real/00170.mp4"
    "2 Celeb-real/id0_0000.mp4"
    "3 Celeb-synthesis/id0_id16_0000.mp4"

But the actual extracted folders are typically:
    YouTube-real/
    Celeb-real/
    Celeb-synthesis/
(without the numeric prefix).

This script normalizes those lines so the saved CSV paths match
the real Windows folders.

Folder mapping used:
- Celeb-synthesis -> fake
- Celeb-real      -> real
- YouTube-real    -> real

Output:
- data/splits/celebdfv2_test.csv

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import os
import csv
import re

DATASET_ROOT = os.path.join("data", "raw", "Celeb-DF-v2")
TEST_LIST = os.path.join(DATASET_ROOT, "List_of_testing_videos.txt")
OUT_CSV = os.path.join("data", "splits", "celebdfv2_test.csv")


def normalize_rel_path(rel_path: str) -> str:
    """
    Removes the leading numeric group prefix:
        "1 YouTube-real/00170.mp4" -> "YouTube-real/00170.mp4"
    Also normalizes slashes.
    """
    text = rel_path.strip().replace("\\", "/")
    text = re.sub(r"^\d+\s+", "", text)  # remove "<digits><space>" at start
    return text


def infer_label_from_path(rel_path: str) -> str:
    rel_lower = rel_path.lower().replace("\\", "/")

    if rel_lower.startswith("celeb-synthesis/"):
        return "fake"

    if rel_lower.startswith("celeb-real/") or rel_lower.startswith("youtube-real/"):
        return "real"

    return "unknown"


def main():
    if not os.path.exists(TEST_LIST):
        raise FileNotFoundError(f"Missing test list file: {TEST_LIST}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    with open(TEST_LIST, "r", encoding="utf-8") as f:
        for line in f:
            raw_rel_path = line.strip()
            if not raw_rel_path:
                continue

            rel_path = normalize_rel_path(raw_rel_path)

            full_path = os.path.join(DATASET_ROOT, rel_path)
            label = infer_label_from_path(rel_path)
            video_id = os.path.splitext(os.path.basename(rel_path))[0]

            rows.append((full_path, label, video_id))

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label", "video_id"])
        for full_path, label, video_id in rows:
            writer.writerow([full_path, label, video_id])

    # Print sanity summary
    exists_count = sum(1 for full_path, _, _ in rows if os.path.exists(full_path))

    print(f"Wrote: {OUT_CSV} ({len(rows)} rows)")
    print(f"Paths that exist on disk: {exists_count}/{len(rows)}")
    for row in rows[:3]:
        print("Sample:", row)


if __name__ == "__main__":
    main()
