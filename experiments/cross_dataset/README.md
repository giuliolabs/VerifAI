# Cross-Dataset Generalization Evaluation

## Overview

This experiment evaluates the **cross-dataset generalization** of the VerifAI deepfake detection model.
A model trained on a benchmark dataset is tested on **unseen, real-world user videos** to assess
robustness and false positive behavior under domain shift.

The evaluation focuses on **real-only data**, measuring how often genuine videos are incorrectly
classified as fake.

---

## Training Dataset

- **FaceForensics++ (C23 compression)**
- Model architecture: **Xception CNN**
- Training performed using standard benchmark splits

---

## Test Dataset

- **Survey369 (real-user recordings)**
- All videos are **authentic (no deepfakes)**
- Videos recorded under uncontrolled conditions:
  - different devices
  - natural lighting
  - handheld motion
  - non-curated framing

This dataset represents a realistic deployment scenario.

---

## Evaluation Setup

- Script: `scripts/eval_cross_dataset.py`
- Input:
  - Trained checkpoint:
    ```
    experiments/ablations/results/ffpp_c23_xception_baseline/best_model.pt
    ```
  - Test CSV:
    ```
    data/splits/survey369_real_test.csv
    ```
- Decision threshold: `prob_fake ≥ 0.5 → fake`

---

## Results (Survey369 – Real-Only)

- **False Positive Rate (FPR):** 0.867
- **Average predicted fake probability:** 0.636
- **Videos evaluated:** 15 real videos
- **Incorrectly flagged as fake:** 13 / 15

Artifacts:
- `survey369_predictions.csv` – per-video predictions
- `metrics_summary.csv` – aggregated metrics
- `cross_test_accuracy.png` – visualization of false positive rate

---

## Interpretation

The high false positive rate indicates **poor cross-dataset generalization**.
The model relies on dataset-specific artifacts learned from FaceForensics++, which do not transfer
well to real-world videos.

This demonstrates a known limitation of deepfake detectors trained solely on curated benchmarks
and highlights the risk of false accusations in real deployment scenarios.

---

## Conclusion

This experiment confirms that strong benchmark performance does not guarantee real-world robustness.
Cross-dataset evaluation is essential for understanding model limitations and deployment risks.

