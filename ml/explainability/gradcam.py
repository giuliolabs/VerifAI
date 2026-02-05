"""
Grad-CAM Explainability Utility (Week 19)
=========================================

Implements Grad-CAM for CNN-based deepfake detectors to visualize
which image regions contribute most to the model prediction.

This implementation is model-agnostic: it can target any Conv2d layer
and computes a heatmap from gradients + activations.

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

Author: Giulio Dajani
Project: VerifAI – Deepfake Detection Framework
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import cv2


@dataclass
class GradCamResult:
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
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def close(self) -> None:
        """Remove hooks (recommended)."""
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()

    def _save_activation(self, module: nn.Module, inputs, output) -> None:
        self._activations = output

    def _save_gradient(self, module: nn.Module, grad_input, grad_output) -> None:
        # grad_output is a tuple; first entry is gradient wrt output of the layer
        self._gradients = grad_output[0]

    @torch.no_grad()
    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
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
        Compute Grad-CAM.

        Args:
            input_tensor: torch tensor shaped [1, 3, H, W]
            original_rgb: original frame as RGB uint8 (H, W, 3) for overlay
            target_class: optional class index (not needed for single-logit models)

        Returns:
            GradCamResult(heatmap, overlay_rgb)
        """
        self.model.zero_grad(set_to_none=True)

        # forward
        output = self.model(input_tensor)

        # output could be [1,1] logit or [1] logit
        if output.ndim == 2 and output.shape[1] == 1:
            score = output[0, 0]
        else:
            score = output.squeeze()[0] if output.ndim > 0 else output

        # backward
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        # activations: [1, C, h, w], gradients: [1, C, h, w]
        activations = self._activations.detach()
        gradients = self._gradients.detach()

        # weights: global average pool gradients over spatial dims
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=False)  # [1, h, w]
        cam = torch.relu(cam)[0].cpu().numpy()  # (h, w)

        # resize heatmap to original image
        H, W = original_rgb.shape[0], original_rgb.shape[1]
        cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
        heatmap = self._normalize_heatmap(cam_resized)

        # overlay: apply color map (OpenCV uses BGR)
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        # blend
        overlay = (0.55 * original_rgb.astype(np.float32) + 0.45 * colored_rgb.astype(np.float32))
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return GradCamResult(heatmap=heatmap, overlay_rgb=overlay)


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Finds the last nn.Conv2d layer in the model (common choice for Grad-CAM).
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model (Grad-CAM requires CNN conv layers).")
    return last_conv
