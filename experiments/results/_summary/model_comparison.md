# Model Comparison Summary

Auto-generated from `experiments/results/*`.

## Executive summary

- **Best overall (by Accuracy):** **fakeavceleb_av_fusion_v1** (FakeAVCeleb) — Type: AV fusion, Backbone: Unknown, Acc=0.9986, BalAcc=N/A, MCC=N/A

### Best model per dataset (by Accuracy)

- **FaceForensics++ C23:** **ffpp_c23_vit_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.7697, BalAcc=N/A, MCC=N/A
- **FakeAVCeleb:** **fakeavceleb_av_fusion_v1** (FakeAVCeleb) — Type: AV fusion, Backbone: Unknown, Acc=0.9986, BalAcc=N/A, MCC=N/A

### Best visual model per dataset (by Accuracy)

- **FaceForensics++ C23:** **ffpp_c23_vit_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.7697, BalAcc=N/A, MCC=N/A
- **FakeAVCeleb:** N/A (no visual models detected)

## Global ranking

### Ranking by Accuracy

1. **fakeavceleb_av_fusion_v1** (FakeAVCeleb) — Type: AV fusion, Backbone: Unknown, Acc=0.9986, BalAcc=N/A, MCC=N/A
2. **fakeavceleb_audio_resnet_baseline** (FakeAVCeleb) — Type: Audio, Backbone: ResNet18, Acc=0.9687, BalAcc=0.6991, MCC=0.6208
3. **fakeavceleb_wav_encoder_baseline** (FakeAVCeleb) — Type: Audio, Backbone: WAV encoder, Acc=0.9480, BalAcc=N/A, MCC=N/A
4. **ffpp_c23_vit_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.7697, BalAcc=N/A, MCC=N/A
5. **ffpp_c23_xception_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: Xception, Acc=0.7458, BalAcc=N/A, MCC=N/A
6. **ffpp_c23_temporal_mobilenetv2** (FaceForensics++ C23) — Type: Visual, Backbone: MobileNetV2, Acc=0.5674, BalAcc=N/A, MCC=N/A
7. **ffpp_c23_mobilenet_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: MobileNetV2, Acc=0.4354, BalAcc=N/A, MCC=N/A
8. **ffpp_c23_temporal_vit** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.3708, BalAcc=0.5442, MCC=0.0927

### Ranking by Balanced Accuracy (if available)

1. **fakeavceleb_audio_resnet_baseline** (FakeAVCeleb) — Type: Audio, Backbone: ResNet18, Acc=0.9687, BalAcc=0.6991, MCC=0.6208
2. **ffpp_c23_temporal_vit** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.3708, BalAcc=0.5442, MCC=0.0927

### Ranking by MCC (if available)

1. **fakeavceleb_audio_resnet_baseline** (FakeAVCeleb) — Type: Audio, Backbone: ResNet18, Acc=0.9687, BalAcc=0.6991, MCC=0.6208
2. **ffpp_c23_temporal_vit** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.3708, BalAcc=0.5442, MCC=0.0927

## Full comparison table

| Model                             | Dataset             | Type      | Backbone    | Acc    | Bal Acc | F1(w)  | F1(m)  | AUC    | AP     | MCC    |
|-----------------------------------|---------------------|-----------|-------------|--------|---------|--------|--------|--------|--------|--------|
| fakeavceleb_audio_resnet_baseline | FakeAVCeleb         | Audio     | ResNet18    | 0.9687 | 0.6991  | 0.9622 | 0.7766 | 0.8544 | 0.9903 | 0.6208 |
| fakeavceleb_av_fusion_v1          | FakeAVCeleb         | AV fusion | Unknown     | 0.9986 | N/A     | 0.9986 | 0.9928 | 0.9997 | 1.0000 | N/A    |
| fakeavceleb_wav_encoder_baseline  | FakeAVCeleb         | Audio     | WAV encoder | 0.9480 | N/A     | 0.9226 | 0.4866 | 0.4958 | 0.9495 | N/A    |
| ffpp_c23_mobilenet_baseline       | FaceForensics++ C23 | Visual    | MobileNetV2 | 0.4354 | N/A     | 0.6776 | 0.4898 | 0.4712 | 0.7511 | N/A    |
| ffpp_c23_temporal_mobilenetv2     | FaceForensics++ C23 | Visual    | MobileNetV2 | 0.5674 | N/A     | 0.5833 | 0.4377 | 0.4216 | 0.7382 | N/A    |
| ffpp_c23_temporal_vit             | FaceForensics++ C23 | Visual    | ViT         | 0.3708 | 0.5442  | 0.3607 | 0.3703 | N/A    | N/A    | 0.0927 |
| ffpp_c23_vit_baseline             | FaceForensics++ C23 | Visual    | ViT         | 0.7697 | N/A     | 0.6695 | 0.4349 | 0.4638 | 0.7472 | N/A    |
| ffpp_c23_xception_baseline        | FaceForensics++ C23 | Visual    | Xception    | 0.7458 | N/A     | 0.7162 | 0.5650 | N/A    | N/A    | N/A    |