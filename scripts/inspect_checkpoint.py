"""
Checkpoint Inspector – Week 19
==============================

This utility script inspects a PyTorch checkpoint (.pt file) and prints
basic structural information about it.

It is useful when:
- verifying that a checkpoint contains the expected parameters
- checking compatibility between a checkpoint and a model architecture
- debugging loading errors (missing or unexpected keys)

The script reports:
- the top-level object type
- top-level dictionary keys (if present)
- the number of tensors in the model state dictionary
- the first set of parameter names (to verify architecture structure)

This inspection step was used during Week 19 to confirm that trained
models match the expected architecture before running Grad-CAM
explainability experiments.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
Required packages:

    pip install torch

------------------------------------------------
RUN
------------------------------------------------
Example usage:

python scripts/inspect_checkpoint.py \
    --checkpoint experiments/results/ffpp_c23_xception_baseline/best_model.pt

------------------------------------------------
OUTPUT
------------------------------------------------
Example output:

    === TOP LEVEL TYPE ===
    <class 'dict'>

    === TOP LEVEL KEYS ===
    model_state
    optimizer_state
    epoch

    === NUM PARAM TENSORS ===
    312

    === FIRST 40 PARAM NAMES ===
    conv1.weight
    bn1.weight
    bn1.bias
    ...

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# -------------------------------------------------------
# Standard Library Imports
# -------------------------------------------------------

# argparse enables command-line arguments
import argparse

# -------------------------------------------------------
# Third-Party Imports
# -------------------------------------------------------

# PyTorch is required to load checkpoint files
import torch


# -------------------------------------------------------
# Helper: Extract state_dict from checkpoint
# -------------------------------------------------------
def extract_state_dict(ckpt):
    """
    Extract the model state dictionary from a checkpoint.

    Different training pipelines store model weights under
    different keys, so we check common patterns.

    Supported formats:
        {"state_dict": ...}
        {"model_state_dict": ...}
        raw state_dict
    """

    if isinstance(ckpt, dict):

        if "state_dict" in ckpt:
            return ckpt["state_dict"]

        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]

    return ckpt


# -------------------------------------------------------
# Main inspection routine
# -------------------------------------------------------
def main():
    """
    Load a checkpoint file and print diagnostic information.
    """

    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch checkpoint file."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint file"
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # -------------------------------------------------------
    # Print top-level object type
    # -------------------------------------------------------
    print("=== TOP LEVEL TYPE ===")
    print(type(ckpt))

    # -------------------------------------------------------
    # If checkpoint is a dictionary, list keys
    # -------------------------------------------------------
    if isinstance(ckpt, dict):

        print("\n=== TOP LEVEL KEYS (first 30) ===")

        for i, k in enumerate(list(ckpt.keys())[:30]):
            print(f"{i+1:02d}. {k}")

    # -------------------------------------------------------
    # Extract state dictionary
    # -------------------------------------------------------
    state = extract_state_dict(ckpt)

    print("\n=== STATE DICT TYPE ===")
    print(type(state))

    # If state_dict is not a mapping
    if not hasattr(state, "keys"):
        print("State is not a mapping. Cannot inspect keys.")
        return

    # -------------------------------------------------------
    # Count parameter tensors
    # -------------------------------------------------------
    keys = list(state.keys())

    print("\n=== NUM PARAM TENSORS ===")
    print(len(keys))

    # -------------------------------------------------------
    # Display first parameter names
    # -------------------------------------------------------
    print("\n=== FIRST 40 PARAM NAMES ===")

    for k in keys[:40]:
        print(k)


# -------------------------------------------------------
# Script entry point
# -------------------------------------------------------
if __name__ == "__main__":
    main()