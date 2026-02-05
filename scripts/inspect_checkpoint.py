"""
Checkpoint Inspector – Week 19
==============================

Prints basic info about a .pt checkpoint:
- top-level keys (if dict)
- number of tensors in the state dict
- first 30 parameter names (so we can match the model architecture)

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

import argparse
import torch


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
    return ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    print("=== TOP LEVEL TYPE ===")
    print(type(ckpt))

    if isinstance(ckpt, dict):
        print("\n=== TOP LEVEL KEYS (first 30) ===")
        for i, k in enumerate(list(ckpt.keys())[:30]):
            print(f"{i+1:02d}. {k}")

    state = extract_state_dict(ckpt)

    print("\n=== STATE DICT TYPE ===")
    print(type(state))

    if not hasattr(state, "keys"):
        print("State is not a mapping. Cannot inspect keys.")
        return

    keys = list(state.keys())
    print("\n=== NUM PARAM TENSORS ===")
    print(len(keys))

    print("\n=== FIRST 40 PARAM NAMES ===")
    for k in keys[:40]:
        print(k)


if __name__ == "__main__":
    main()
