# VerifAI - AI-Powered Deepfake Detection Platform

## Live Demo

**Website:** https://vrifai.com — runs entirely in your browser (ONNX Runtime Web); videos are never uploaded.

> Hosted for free on GitHub Pages (`website/`). The in-browser models are ONNX exports of the final checkpoints (`scripts/export_onnx.py`). See [DEPLOY.md](DEPLOY.md).

---

## Overview

VerifAI is a **production-ready deepfake detection platform** capable of analyzing uploaded videos and determining whether they are authentic or AI-manipulated.

The system supports:

- Visual-only deepfake detection  
- Audio-visual deepfake detection  
- Automatic modality detection
- Multimodal late fusion architecture  
- Real-time API-based inference  
- Probability-based decision output  

This project combines **academic research methodology** with **production engineering practices**.

---

## How to Use (Web Demo)

1. Open the live demo:  
   https://vrifai.com  

2. Drop a video onto the upload box (or click **browse**)

3. Choose a short video file:
   - `.mp4`
   - `.mov`
   - `.avi`

The video may:
- Contain audio  
- Have no audio track  
- Contain fake video  
- Contain fake audio  
- Contain both  

4. Tick the consent box and click **Analyse video**

5. Wait a few seconds while the system processes the upload.

6. View the result.

---

## Output Format

Example response:

```json
{
  "label": "fake",
  "prob_fake": "93.41",
  "mode": "hybrid_av"
}
```
## Output Explanation

### `label`

- `"real"` → predicted authentic  
- `"fake"` → predicted manipulated  

---

### `prob_fake`

Probability (in %) that the video is fake.

Example: `93.41` means **93.41% confidence** the video is fake.

---

### `mode`

Indicates which detection pipeline was used:

- `hybrid_av` → Multimodal fusion model (video + audio)  
- `visual_only_fallback` → Vision-only fallback model (no usable audio detected)  

The system automatically selects the appropriate model based on audio presence.

---

# System Architecture

## Inference Pipeline

- Extract 5 representative video frames  
- Extract audio track (if present)  
- Compute MFCC audio features  
- Detect if audio contains usable signal  
- Route to correct model:

  - **Video + Audio → Multimodal Fusion Model**
  - **Video only → Vision Transformer (ViT) fallback**

---

## Multimodal Fusion Model

### Video Branch

- MobileNetV2 backbone (ImageNet pretrained)  
- Frame-level embedding  
- Temporal averaging across frames  

### Audio Branch

- ResNet18 backbone adapted for 1-channel MFCC input  

### Fusion

- Concatenation of video + audio embeddings  
- MLP classifier  
- BCEWithLogitsLoss  
- Sigmoid → probability output  

---

## Visual Fallback Model

- Vision Transformer (ViT)  
- Multi-frame scoring  
- Logit averaging across frames  
- Stable probability estimation  

---

# Tech Stack

## Backend

- FastAPI  
- Uvicorn  

## Deep Learning

- PyTorch  
- Torchvision  
- timm (Vision Transformer)  

## Audio Processing

- Librosa  
- MFCC feature extraction  

## Video Processing

- OpenCV  
- MoviePy  

## Deployment

- In-browser inference: ONNX Runtime Web (WebAssembly), int8-compressed ONNX models  
- GitHub Pages + custom domain (vrifai.com)  
- FastAPI backend for local/API use  

---

# Datasets Used

## FakeAVCeleb_v1.2

- Type: multimodal (audio + video)  
- Used for multimodal fusion training  
- Includes:
  - FakeVideo-FakeAudio  
  - FakeVideo-RealAudio  
  - RealVideo-FakeAudio  
  - RealVideo-RealAudio  

## FaceForensics++ (C23)

- Used for visual baseline training  

## Celeb-DF-v2

- High-quality visual deepfake benchmark  

## DeeperForensics

- Robustness testing under perturbations  

## Survey369

- University-approved real dataset  
- Real-only evaluation  
- GDPR compliant  
- Not publicly redistributed  

---

# Engineering Highlights

- Automatic audio detection  
- Multi-frame inference stabilization  
- Subject-level dataset splitting (prevents leakage)  
- Per-category evaluation metrics  
- Balanced accuracy reporting  
- Secure file validation  
- MP4 signature verification  
- Upload size limits  
- Temporary file cleanup  
- Modular ML pipeline  
- Research + production hybrid design  

---

# Project Structure

```bash
VerifAI/
│
├── backend/        # FastAPI inference API
├── ml/             # Models, training, evaluation
├── pipelines/      # Dataset preprocessing
├── data/           # Raw and processed datasets
├── experiments/    # Checkpoints and logs
├── scripts/        # Utilities
├── docs/           # Weekly progress reports and diagrams
└── venv
```

---

# Run Locally

## 1️. Clone repository

```bash
git clone https://github.com/giuliolabs/VerifAI.git
cd VerifAI
```

## 2️. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

## 3️. Install dependencies

```bash
pip install -r requirements.txt
```

## 4️. Start API

```bash
uvicorn backend.main:app --reload
```

Open browser:  
http://127.0.0.1:8000

---

# Evaluation Metrics

The system is evaluated using:

- Accuracy  
- Balanced Accuracy  
- ROC-AUC  
- F1 Score  
- MCC  
- Confusion Matrix  
- Per-category performance breakdown  

Subject-level splits are used to prevent data leakage and overfitting.

---

# Roadmap (v2.0)

Planned future improvements:

- Cross-modal attention fusion  
- Temporal transformer modeling  
- Docker containerization  
- CI/CD integration  
- AWS production deployment  
- Domain generalization improvements  

---

# Author

**Giulio Dajani**  
AI Engineer | Software Engineer  
University of Greenwich
