# KoDF Usage Note (Week 19)

## Background
KoDF (Korean Deepfake Detection Dataset) is a large-scale deepfake dataset
introduced by Kwon et al. (ICCV 2021), containing over 60,000 real videos and
175,000 synthesized videos generated using multiple deepfake methods.

## Intended Role in This Project
KoDF is referenced as a **source training dataset** to motivate cross-dataset
generalization experiments. Models used in this project were originally trained
using KoDF-style data distributions, following established deepfake detection
literature.

## Practical Constraint
The full KoDF dataset exceeds **500 GB uncompressed**. Due to hardware and
storage limitations on a personal laptop, the complete video corpus was not
re-downloaded or re-extracted during Week 19.

## Experimental Decision
Instead of re-ingesting KoDF locally, **cross-dataset generalization was evaluated
by testing trained models on Celeb-DF-v2**, a standard unseen-domain benchmark.

## Reference
Kwon, P. et al., *KoDF: A Large-Scale Korean DeepFake Detection Dataset*,
ICCV 2021.
