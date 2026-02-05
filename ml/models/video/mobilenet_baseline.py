"""
MobileNetV2 Baseline (Binary Classifier)
Dependencies:
    pip install torch torchvision
    pip install torch torchvision torchaudio
    pip install scikit-learn tqdm pillow numpy
"""

import torch.nn as nn
from torchvision import models


def build_mobilenet_v2_binary():
    """
    Returns a MobileNetV2 model with a 1-logit binary classifier head.
    Output logits shape: [batch]
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model
