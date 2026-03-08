# Model Comparison Summary

Auto-generated from `experiments/results/*`.

## Executive summary

- **Best overall (by Accuracy):** **fakeavceleb_av_fusion_v1** (FakeAVCeleb) — Type: AV fusion, Backbone: Unknown, Acc=0.9981, BalAcc=N/A, MCC=0.9646

### Best model per dataset (by Accuracy)

- **FaceForensics++ C23:** **ffpp_c23_vit_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.7697, BalAcc=N/A, MCC=N/A
- **FakeAVCeleb:** **fakeavceleb_av_fusion_v1** (FakeAVCeleb) — Type: AV fusion, Backbone: Unknown, Acc=0.9981, BalAcc=N/A, MCC=0.9646

### Best visual model per dataset (by Accuracy)

- **FaceForensics++ C23:** **ffpp_c23_vit_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.7697, BalAcc=N/A, MCC=N/A
- **FakeAVCeleb:** N/A (no visual models detected)

## Global ranking

### Ranking by Accuracy

1. **fakeavceleb_av_fusion_v1** (FakeAVCeleb) — Type: AV fusion, Backbone: Unknown, Acc=0.9981, BalAcc=N/A, MCC=0.9646
2. **final_hybrid_av** (FakeAVCeleb) — Type: Unknown, Backbone: Xception, Acc=0.9905, BalAcc=0.9952, MCC=0.8411
3. **fakeavceleb_audio_resnet_baseline** (FakeAVCeleb) — Type: Audio, Backbone: ResNet18, Acc=0.9687, BalAcc=0.6991, MCC=0.6208
4. **fakeavceleb_wav_encoder_baseline** (FakeAVCeleb) — Type: Audio, Backbone: WAV encoder, Acc=0.9480, BalAcc=N/A, MCC=N/A
5. **ffpp_c23_vit_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.7697, BalAcc=N/A, MCC=N/A
6. **ffpp_c23_xception_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: Xception, Acc=0.7458, BalAcc=N/A, MCC=N/A
7. **ffpp_c23_temporal_mobilenetv2** (FaceForensics++ C23) — Type: Visual, Backbone: MobileNetV2, Acc=0.5674, BalAcc=N/A, MCC=N/A
8. **ffpp_c23_mobilenet_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: MobileNetV2, Acc=0.4354, BalAcc=N/A, MCC=N/A
9. **ffpp_c23_temporal_vit** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.3708, BalAcc=0.5442, MCC=0.0927
10. **final_visual_all_datasets_xception** (FaceForensics++ C23) — Type: Visual, Backbone: Xception, Acc=N/A, BalAcc=N/A, MCC=N/A

### Ranking by Balanced Accuracy (if available)

1. **final_hybrid_av** (FakeAVCeleb) — Type: Unknown, Backbone: Xception, Acc=0.9905, BalAcc=0.9952, MCC=0.8411
2. **fakeavceleb_audio_resnet_baseline** (FakeAVCeleb) — Type: Audio, Backbone: ResNet18, Acc=0.9687, BalAcc=0.6991, MCC=0.6208
3. **ffpp_c23_temporal_vit** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.3708, BalAcc=0.5442, MCC=0.0927

### Ranking by MCC (if available)

1. **fakeavceleb_av_fusion_v1** (FakeAVCeleb) — Type: AV fusion, Backbone: Unknown, Acc=0.9981, BalAcc=N/A, MCC=0.9646
2. **final_hybrid_av** (FakeAVCeleb) — Type: Unknown, Backbone: Xception, Acc=0.9905, BalAcc=0.9952, MCC=0.8411
3. **fakeavceleb_audio_resnet_baseline** (FakeAVCeleb) — Type: Audio, Backbone: ResNet18, Acc=0.9687, BalAcc=0.6991, MCC=0.6208
4. **ffpp_c23_temporal_vit** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.3708, BalAcc=0.5442, MCC=0.0927

## Full comparison table

| Model | Dataset | Type | Backbone | Acc | Bal Acc | F1(w) | F1(m) | AUC | AP | MCC |
|------|---------|------|----------|-----|---------|-------|-------|-----|----|-----|
| fakeavceleb_audio_resnet_baseline | FakeAVCeleb | Audio | ResNet18 | 0.9687 | 0.6991 | 0.9622 | 0.7766 | 0.8544 | 0.9903 | 0.6208 |
| fakeavceleb_av_fusion_v1 | FakeAVCeleb | AV fusion | Unknown | 0.9981 | N/A | 0.9981 | 0.9820 | 0.9991 | 1.0000 | 0.9646 |
| fakeavceleb_wav_encoder_baseline | FakeAVCeleb | Audio | WAV encoder | 0.9480 | N/A | 0.9226 | 0.4866 | 0.4958 | 0.9495 | N/A |
| ffpp_c23_mobilenet_baseline | FaceForensics++ C23 | Visual | MobileNetV2 | 0.4354 | N/A | 0.6776 | 0.4898 | 0.4712 | 0.7511 | N/A |
| ffpp_c23_temporal_mobilenetv2 | FaceForensics++ C23 | Visual | MobileNetV2 | 0.5674 | N/A | 0.5833 | 0.4377 | 0.4216 | 0.7382 | N/A |
| ffpp_c23_temporal_vit | FaceForensics++ C23 | Visual | ViT | 0.3708 | 0.5442 | 0.3607 | 0.3703 | N/A | N/A | 0.0927 |
| ffpp_c23_vit_baseline | FaceForensics++ C23 | Visual | ViT | 0.7697 | N/A | 0.6695 | 0.4349 | 0.4638 | 0.7472 | N/A |
| ffpp_c23_xception_baseline | FaceForensics++ C23 | Visual | Xception | 0.7458 | N/A | 0.7162 | 0.5650 | N/A | N/A | N/A |
| final_hybrid_av | FakeAVCeleb | Unknown | Xception | 0.9905 | 0.9952 | 0.9913 | 0.9142 | 0.9956 | 0.9999 | 0.8411 |
| final_visual_all_datasets_xception | FaceForensics++ C23 | Visual | Xception | N/A | N/A | 0.8902 | 0.8530 | N/A | N/A | N/A |