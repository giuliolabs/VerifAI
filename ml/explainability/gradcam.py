"""
Grad-CAM Explainability Utility (Week 19)
=========================================

This module implements a Grad-CAM (Gradient-weighted Class Activation Mapping)
utility for CNN-based deepfake detection models. Its purpose is to provide a
visual explanation of which image regions contributed most strongly to the
model's prediction.

The implementation is designed to be model-agnostic, meaning it can be applied
to any convolutional neural network as long as a suitable Conv2d target layer
is selected. This makes it reusable across different CNN backbones used within
the VerifAI framework.

------------------------------------------------
DEPENDENCIES
------------------------------------------------
    pip install torch numpy opencv-python
------------------------------------------------
OUTPUT
------------------------------------------------
Returns:
- heatmap (H, W) float32 in [0, 1]
- overlay (H, W, 3) uint8 RGB image suitable for saving
------------------------------------------------
NOTES FOR EXAMINERS
------------------------------------------------
- Grad-CAM highlights regions that most influence the model score.
- Heatmaps are qualitative and do not prove manipulation on their own.
- Used in Week 19 to support explainability visuals.

Author: Giulio Dajani 001343717
Project: VerifAI – Deepfake Detection Framework
Copyright © 2026 Giulio Labs
"""

from __future__ import annotations

# dataclass is used to return explainability outputs in a clean structured form
from dataclasses import dataclass

# Optional is used for attributes that may temporarily be None before hooks run
from typing import Optional

# OpenCV is used for resizing heatmaps and creating colored overlays
import cv2

# NumPy is used for numerical processing and image array handling
import numpy as np

# PyTorch is used for gradients, activations, and tensor operations
import torch
import torch.nn as nn


@dataclass
class GradCamResult:
    """
    Structured output of the Grad-CAM process.

    heatmap:
        Normalised intensity map showing which regions influenced the prediction.

    overlay_rgb:
        Original image blended with the colored heatmap for easier interpretation.
    """

    heatmap: np.ndarray  # (H, W) float32 in [0,1]
    overlay_rgb: np.ndarray  # (H, W, 3) uint8 RGB


class GradCAM:
    """
    Simple Grad-CAM implementation for binary logit models.

    Usage:
        cam = GradCAM(model, target_layer)
        result = cam(image_tensor, target_class=None)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        # Store model and the chosen convolutional layer to analyze
        self.model = model
        self.target_layer = target_layer

        # These will hold the feature maps and gradients captured by hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks on the target layer:
        # - forward hook saves activations
        # - backward hook saves gradients
        # Grad-CAM requires both to compute the class activation map
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def close(self) -> None:
        """
        Remove registered hooks.

        This is recommended after use to avoid keeping unnecessary hooks
        attached to the model, especially if Grad-CAM is run repeatedly.
        """
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()

    def _save_activation(self, module: nn.Module, inputs, output) -> None:
        """
        Save the forward activations produced by the target layer.
        These represent the spatial feature maps used later in Grad-CAM.
        """
        self._activations = output

    def _save_gradient(self, module: nn.Module, grad_input, grad_output) -> None:
        """
        Save gradients flowing back through the target layer.

        Grad-CAM uses these gradients to estimate how important each
        feature map channel is for the final prediction.
        """

        # grad_output is returned as a tuple;
        # the first entry corresponds to the gradient with respect to layer output
        self._gradients = grad_output[0]

    @torch.no_grad()
    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Normalize heatmap values into the range [0, 1].

        Negative values are clipped because Grad-CAM focuses on positive
        evidence supporting the model's decision.
        """
        heatmap = np.maximum(heatmap, 0)

        max_val = float(np.max(heatmap)) if heatmap.size else 0.0
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.astype(np.float32)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        original_rgb: np.ndarray,
        target_class: Optional[int] = None,
    ) -> GradCamResult:
        """
        Compute Grad-CAM for a single input image.

        Args:
            input_tensor: torch tensor shaped [1, 3, H, W]
            original_rgb: original frame as RGB uint8 (H, W, 3) for overlay
            target_class: optional class index (not needed for single-logit models)

        Returns:
            GradCamResult containing:
            - normalized heatmap
            - blended RGB overlay
        """

        # Clear any existing gradients before running a new explanation pass
        self.model.zero_grad(set_to_none=True)

        # Forward pass through the model
        output = self.model(input_tensor)

        # Handle different output shapes:
        # some binary classifiers return [1,1], others may return [1]
        # The aim is to isolate the scalar score used for backpropagation
        if output.ndim == 2 and output.shape[1] == 1:
            score = output[0, 0]
        else:
            score = output.squeeze()[0] if output.ndim > 0 else output

        # Backward pass from the selected score
        # This computes gradients needed for Grad-CAM weighting
        score.backward(retain_graph=False)

        # Ensure the hooks successfully captured required information
        if self._activations is None or self._gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        # Activations and gradients are expected in shape [1, C, h, w]
        activations = self._activations.detach()
        gradients = self._gradients.detach()

        # Compute channel importance weights by averaging gradients across the spatial dimensions
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Combine weights with activations to form the class activation map
        cam = (weights * activations).sum(dim=1, keepdim=False)  # [1, h, w]

        # Apply ReLU so only positive contributions are kept
        cam = torch.relu(cam)[0].cpu().numpy()  # (h, w)

        # Resize heatmap to match the original image resolution
        H, W = original_rgb.shape[0], original_rgb.shape[1]
        cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

        # Normalize the resized heatmap to [0,1]
        heatmap = self._normalize_heatmap(cam_resized)

        # Convert heatmap to colored representation
        # OpenCV applies color maps in BGR, so convert back to RGB afterward
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        # Blend the heatmap with the original image so the explanation
        # remains visually interpretable instead of replacing the frame entirely
        overlay = (0.55 * original_rgb.astype(np.float32) + 0.45 * colored_rgb.astype(np.float32))
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return GradCamResult(heatmap=heatmap, overlay_rgb=overlay)


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Find the last convolutional layer in a CNN.

    This is a common default choice for Grad-CAM because deeper convolutional
    layers usually capture higher-level semantic features while still preserving
    some spatial information.
    """

    last_conv = None

    # Iterate through all submodules and keep updating the reference
    # whenever a Conv2d layer is found
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    # Grad-CAM requires at least one convolutional layer
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model (Grad-CAM requires CNN conv layers).")

    return last_conv