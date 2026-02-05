# Week 19 – Explainability and Cross-Dataset Generalization (FF++ → Celeb-DF-v2)

## Overview
Week 19 focuses on two advanced evaluation aspects of the VerifAI framework:

1. **Model explainability**, using Grad-CAM heatmaps to visualize spatial regions
   contributing to deepfake predictions.
2. **Cross-dataset generalization**, evaluating how a model trained on one dataset
   performs on an unseen dataset.

These experiments address robustness, transparency, and trustworthiness of
deepfake detection models, as highlighted in recent literature (e.g. Lyu, 2020).

---

## Explainability: Offline Grad-CAM

### Motivation
Deepfake detection models are often criticized for being black boxes.
Explainability techniques help verify whether a model focuses on meaningful
regions (e.g. facial landmarks, mouth area, eyes) rather than spurious cues.

Grad-CAM (Gradient-weighted Class Activation Mapping) was selected due to its
simplicity, interpretability, and wide adoption in CNN-based vision models.

### Implementation
- Model: **Xception CNN (frame-level)**
- Source: `ml/models/video/xception.py`
- Explainability module: `ml/explainability/gradcam.py`
- Execution script: `scripts/run_gradcam.py`

For each selected sample video:
1. A representative middle frame is extracted using OpenCV.
2. The frame is passed through the trained Xception model.
3. Grad-CAM computes gradients with respect to the last convolutional layer.
4. A heatmap is generated and overlaid on the original frame.
