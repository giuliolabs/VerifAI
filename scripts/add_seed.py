from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import random

ROOT = Path(__file__).resolve().parents[1]
SEEDS_FILE = ROOT / "experiments" / "logs" / "verifai_random_seeds.tsv"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", required=True, help="Short note describing what this seed is used for")
    parser.add_argument("--seed", type=int, default=None, help="Optional explicit seed")
    args = parser.parse_args()

    SEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randint(1, 2_147_483_647)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = SEEDS_FILE.exists()
    with SEEDS_FILE.open("a", encoding="utf-8") as f:
        if not file_exists:
            f.write("seed\tnote\ttimestamp\n")
        f.write(f"{seed}\t{args.note}\t{stamp}\n")

    print(f"[OK] Added seed: {seed} -> {SEEDS_FILE}")

if __name__ == "__main__":
    main()
