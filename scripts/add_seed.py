"""
Random Seed Registry Utility – VerifAI
=====================================

This script records random seeds used during experiments in a central
registry file to improve reproducibility and traceability of results.

In machine learning experiments, random seeds affect several processes:

    - Model weight initialization
    - Data shuffling
    - Augmentation randomness
    - Train/validation split generation

By logging seeds alongside a short note describing their purpose,
researchers can reproduce experiments more reliably.

------------------------------------------------
OUTPUT FILE
------------------------------------------------

Seeds are recorded in:

    experiments/logs/verifai_random_seeds.tsv

The TSV file contains:

    seed    note    timestamp

Example entry:

    183742910    xception_train_ffpp_seed    2026-02-22 14:51:03

------------------------------------------------
USAGE
------------------------------------------------

Generate a random seed automatically:

    python scripts/register_seed.py --note "xception training run"

Provide an explicit seed:

    python scripts/register_seed.py --seed 12345 --note "cross-dataset evaluation"

------------------------------------------------
WHY THIS EXISTS
------------------------------------------------

Reproducibility is a key requirement in machine learning research.
Recording seeds ensures that training runs, evaluation experiments,
and ablation studies can be reproduced exactly if required.

This utility is part of VerifAI's reproducibility infrastructure
alongside:

    - experiment logs
    - run metadata
    - training curves
    - debug logs

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# -------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------

# argparse handles command-line arguments
import argparse

# pathlib provides cross-platform file system paths
from pathlib import Path

# datetime is used to timestamp each seed entry
from datetime import datetime

# random generates seeds when one is not explicitly provided
import random


# -------------------------------------------------------
# Project paths
# -------------------------------------------------------

# Root of the repository (parent of /scripts directory)
ROOT = Path(__file__).resolve().parents[1]

# Location of the seed registry file
SEEDS_FILE = ROOT / "experiments" / "logs" / "verifai_random_seeds.tsv"


# -------------------------------------------------------
# Main logic
# -------------------------------------------------------

def main() -> None:
    """
    Register a random seed in the VerifAI seed registry.

    The script either:
        - uses a provided seed value
        - generates a random seed automatically

    The seed is then appended to the TSV registry file.
    """

    # ---------------------------------------------
    # Parse command-line arguments
    # ---------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--note",
        required=True,
        help="Short description of what the seed will be used for"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional explicit seed value"
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Ensure the logs directory exists
    # ---------------------------------------------
    SEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------
    # Determine seed value
    # ---------------------------------------------
    if args.seed is not None:
        seed = args.seed
    else:
        # Generate a random 32-bit positive integer seed
        seed = random.randint(1, 2_147_483_647)

    # Create timestamp
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------------------------------------
    # Append seed to registry file
    # ---------------------------------------------
    file_exists = SEEDS_FILE.exists()

    with SEEDS_FILE.open("a", encoding="utf-8") as f:

        # Write header if file is new
        if not file_exists:
            f.write("seed\tnote\ttimestamp\n")

        # Write entry
        f.write(f"{seed}\t{args.note}\t{stamp}\n")

    # ---------------------------------------------
    # Confirmation output
    # ---------------------------------------------
    print(f"[OK] Added seed: {seed} -> {SEEDS_FILE}")


# -------------------------------------------------------
# Script entry point
# -------------------------------------------------------

if __name__ == "__main__":
    main()