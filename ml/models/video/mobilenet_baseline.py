"""
MobileNetV2 Baseline (Binary Classifier)
========================================

This module defines a lightweight MobileNetV2 baseline model for
binary deepfake detection. It uses an ImageNet-pretrained backbone
and replaces the classification head with a single-logit output
layer suitable for real vs fake prediction.

MobileNetV2 was selected as a baseline due to:
- Low computational cost
- Strong performance on image classification tasks
- Suitability for frame-based deepfake detection

Dependencies:
    pip install torch torchvision
    pip install scikit-learn tqdm pillow numpy

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

# PyTorch neural network components
import torch.nn as nn

# torchvision provides pretrained MobileNetV2
from torchvision import models


def build_model(**kwargs):
    """
    Convenience wrapper to maintain consistency with other
    model builder interfaces inside VerifAI.

    This allows the model to be registered and accessed
    via the video model registry.
    """
    return build_mobilenet_v2_binary(**kwargs)


def build_mobilenet_v2_binary():
    """
    Build MobileNetV2 adapted for binary classification.

    Returns:
        model with output logit shape [batch, 1]

    The original ImageNet classifier head (1000 classes)
    is replaced with a single-output linear layer.
    """

    # Load pretrained MobileNetV2 backbone.
    # Pretraining improves convergence and generalization.
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    # Extract number of input features to the classifier layer.
    # This ensures compatibility when replacing the final layer.
    in_features = model.classifier[1].in_features

    # Replace the final classification layer with a single logit output.
    # A single logit is preferred for binary classification,
    # allowing use of BCEWithLogitsLoss during training.
    model.classifier[1] = nn.Linear(in_features, 1)

    return model