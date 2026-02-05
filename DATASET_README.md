# DATASET_README

## Datasets Used
This project uses the following datasets for deepfake detection research, benchmarking, and evaluation:

1) Celeb-DF-v2
- Type: real/fake videos (high visual quality)
- Use: primary benchmark dataset for state-of-the-art deepfake detection
- Stored at: data\raw\Celeb-DF-v2
- Notes: widely used academic benchmark; original train/test split respected where applicable.

2) DeeperForensics
- Type: real/fake videos with diverse perturbations
- Use: robustness evaluation under real-world degradations (blur, noise, compression, occlusion)
- Stored at: data\raw\DeeperForensics
- Notes: used to evaluate generalisation beyond standard deepfake artifacts.

3) FaceForensics++_C23
- Type: real/fake videos with multiple manipulation methods
- Use: baseline model training and controlled evaluation (Pan et al. style setup)
- Stored at: data\raw\FaceForensics++_C23
- Notes:
  - Compression level: C23 (medium compression)
  - Manipulations include Deepfakes, Face2Face, FaceSwap, NeuralTextures
  - Frames are sampled per video for CNN-based baselines.

4) FakeAVCeleb_v1.2
- Type: multimodal (audio/video) deepfake dataset
- Labels: ARVR / AFVR / ARVF / AFVF
- Use: audio-visual fusion experiments and modality-specific analysis
- Stored at: data\raw\FakeAVCeleb_v1.2
- Notes: modality labels preserved for per-modality metrics and ablation studies.

5) Survey369 (University-approved real-only dataset)
- Type: real images/videos from consenting participants
- Use:
  - in-the-wild real-domain adaptation
  - final real-only holdout evaluation
- Stored at: data\raw\Survey369
- Restrictions:
  - contains personal data
  - not redistributed
  - stored securely
  - GDPR compliant
- Notes: used exclusively for evaluation and domain calibration, never for fake synthesis.

## Storage & Security
- All raw datasets are stored immutably under `data/raw/`.
- Any extracted frames, crops, or derived artifacts are stored under `data/processed/`.
- Survey369 is encrypted at rest and never pushed to public repositories.
- Dataset splits are performed at the video or subject level to prevent data leakage.

## Licensing / Restrictions
- Celeb-DF-v2, DeeperForensics, FaceForensics++, FakeAVCeleb:
  - used strictly under their respective published academic licenses.
- Survey369:
  - internal academic research use only
  - access restricted to approved researchers.

## Contact
Dataset management and compliance:
Giulio Dajani 001343717
University of Greenwich
