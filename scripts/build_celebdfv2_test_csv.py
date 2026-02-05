"""
Celeb-DF-v2 Test CSV Builder (Week 19)
=====================================

Builds a CSV manifest for Celeb-DF-v2 evaluation using the official
List_of_testing_videos.txt file included with the dataset.

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

DATASET_ROOT = os.path.join("data", "raw", "Celeb-DF-v2")
TEST_LIST = os.path.join(DATASET_ROOT, "List_of_testing_videos.txt")
OUT_CSV = os.path.join("data", "splits", "celebdfv2_test.csv")


def infer_label_from_path(rel_path: str) -> str:
    rel_lower = rel_path.lower().replace("\\", "/")

    if "celeb-synthesis" in rel_lower:
        return "fake"

    # Handles: "Celeb-real", "YouTube-real", and also "1 YouTube-real"
    if "celeb-real" in rel_lower or "youtube-real" in rel_lower:
        return "real"

    return "unknown"



def main():
    if not os.path.exists(TEST_LIST):
        raise FileNotFoundError(f"Missing test list file: {TEST_LIST}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    with open(TEST_LIST, "r", encoding="utf-8") as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue

            # The list file typically stores relative paths like:
            # Celeb-synthesis/id0_id16_0000.mp4
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

    print(f"Wrote: {OUT_CSV} ({len(rows)} rows)")
    # quick sanity: print first 3
    for row in rows[:3]:
        print("Sample:", row)


if __name__ == "__main__":
    main()
