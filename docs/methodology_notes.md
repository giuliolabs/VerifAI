# Methodology

## 1. Research Design Overview

This project follows a **design science and experimental research methodology**, combining
software engineering principles with applied machine learning experimentation.
The goal is to design, implement, and evaluate a **context-aware, multimodal deepfake
detection and verification framework** that is accurate, explainable, and deployable
as a real-world web service.

The methodology is structured into modular phases, allowing individual components
(e.g. data pipelines, detection models, explainability modules) to be developed,
tested, and evaluated independently. This modularity supports both academic
rigor and long-term system extensibility.

---

## 2. Data Sources and Ethical Compliance

### 2.1 Public Datasets

The system is trained and evaluated using three well-established deepfake datasets:

- **DFDC (Deepfake Detection Challenge)** – large-scale, diverse video dataset.
- **FakeAVCeleb** – multimodal dataset containing synchronized fake audio and video.
- **KoDF (Korean DeepFake Dataset)** – demographically diverse dataset addressing ethnic bias.

These datasets were selected to ensure coverage of:
- multiple manipulation techniques,
- diverse demographics,
- varying video and audio qualities.

### 2.2 User-Contributed Dataset

In addition to public datasets, a **custom dataset of 369 real videos and images**
was collected through an ethics-approved survey. All participants provided informed
consent, and data handling complies with GDPR and University of Greenwich ethical guidelines.
These samples are used exclusively as *authentic (real)* data to improve generalization
and realism.

---

## 3. Data Pipeline and Preprocessing

A reproducible, multi-stage data pipeline was implemented to ensure consistency
across datasets.

### 3.1 Pipeline Stages

1. **Integrity Validation**  
   Corrupted or incomplete media files are detected and excluded.

2. **Video Processing**  
   - Videos are converted to a standard format (MP4, consistent resolution).
   - Frames are sampled at fixed intervals to reduce redundancy.

3. **Audio Processing**  
   - Audio tracks are extracted and converted to WAV format.
   - MFCC features are computed to capture vocal characteristics.

4. **Metadata Extraction**  
   - EXIF and ffmpeg metadata are parsed (timestamps, codecs, software tags).
   - Social-media style compression indicators are logged when present.
   - The pipeline is designed to detect emerging standards such as C2PA manifests.

5. **Dataset Splitting**  
   Data is split into training, validation, and test sets using subject-level separation
to avoid identity leakage. A real-world holdout set is reserved for final evaluation.

---

## 4. Model Architecture

### 4.1 Visual Stream

Video frames are processed using pre-trained CNN and transformer-based models
(e.g. Xception, Vision Transformers), fine-tuned on the prepared datasets.
Temporal consistency is captured using sequence models (LSTM or temporal transformers).

### 4.2 Audio Stream

Audio features (MFCCs) are processed using CNN-based encoders to detect artifacts
introduced by voice cloning or synthesis.

### 4.3 Multimodal Fusion

Audio and visual features are fused using a dedicated multimodal fusion module.
This enables detection of cross-modal inconsistencies, such as mismatched lip
movements and speech patterns.

---

## 5. Explainability and Context Awareness

To address the black-box nature of deep learning models, the system integrates
multiple explainability mechanisms:

- **Grad-CAM visualizations** highlighting manipulated facial regions.
- **Frame-level confidence logs** for temporal transparency.
- **Text-based saliency reports** explaining contributing features.
- **Metadata-based provenance analysis** to contextualize predictions.

These outputs are designed to be understandable by non-expert users, supporting
trust and usability in journalistic, legal, and public-facing contexts.

---

## 6. System Implementation

- **Backend:** FastAPI (Python), PyTorch, OpenCV, librosa  
- **Frontend:** Next.js (React), deployed on Vercel  
- **Database:** PostgreSQL for users, subscriptions, and audit logs  
- **Deployment:** Dockerized services deployed on AWS EC2 with GPU support  

A CI/CD pipeline automates testing, container builds, and deployment.

---

## 7. Evaluation Strategy

Model performance is evaluated using:
- Accuracy, Precision, Recall, F1-score, and AUC
- Cross-dataset generalization tests
- Real-world evaluation on unseen, user-contributed content

Usability and trust are evaluated through survey feedback and interaction metrics.

---

## 8. Summary

This methodology ensures that the proposed framework is:
- scientifically rigorous,
- ethically compliant,
- modular and extensible,
- suitable for real-world deployment.

By combining multimodal detection, explainability, and contextual analysis,
the system aims to advance both academic research and practical deepfake verification.
