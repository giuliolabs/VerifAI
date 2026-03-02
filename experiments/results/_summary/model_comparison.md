# Model Comparison Summary

Auto-generated from `experiments/results/*`.

## Executive summary

- **Best overall (by Accuracy):** **fakeavceleb_av_fusion_v1** (FakeAVCeleb) — Type: AV fusion, Backbone: Unknown, Acc=0.9990, BalAcc=N/A, MCC=N/A

### Best model per dataset (by Accuracy)

- **FaceForensics++ C23:** **ffpp_c23_xception_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: Xception, Acc=0.7458, BalAcc=N/A, MCC=N/A
- **FakeAVCeleb:** **fakeavceleb_av_fusion_v1** (FakeAVCeleb) — Type: AV fusion, Backbone: Unknown, Acc=0.9990, BalAcc=N/A, MCC=N/A

### Best visual model per dataset (by Accuracy)

- **FaceForensics++ C23:** **ffpp_c23_xception_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: Xception, Acc=0.7458, BalAcc=N/A, MCC=N/A
- **FakeAVCeleb:** N/A (no visual models detected)

## Global ranking

### Ranking by Accuracy

1. **fakeavceleb_av_fusion_v1** (FakeAVCeleb) — Type: AV fusion, Backbone: Unknown, Acc=0.9990, BalAcc=N/A, MCC=N/A
2. **fakeavceleb_audio_resnet_baseline** (FakeAVCeleb) — Type: Audio, Backbone: ResNet18, Acc=0.9672, BalAcc=N/A, MCC=N/A
3. **fakeavceleb_wav_encoder_baseline** (FakeAVCeleb) — Type: Audio, Backbone: WAV encoder, Acc=0.9480, BalAcc=0.5000, MCC=0.0000
   - High Acc but low BalAcc (possible class imbalance bias)
   - MCC≈0 (weak correlation / unreliable)
4. **ffpp_c23_xception_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: Xception, Acc=0.7458, BalAcc=N/A, MCC=N/A
5. **ffpp_c23_vit_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.6871, BalAcc=N/A, MCC=N/A
6. **ffpp_c23_temporal_mobilenetv2** (FaceForensics++ C23) — Type: Visual, Backbone: MobileNetV2, Acc=0.5674, BalAcc=0.4348, MCC=-0.1210
7. **ffpp_c23_mobilenet_baseline** (FaceForensics++ C23) — Type: Visual, Backbone: MobileNetV2, Acc=0.4087, BalAcc=N/A, MCC=N/A
8. **ffpp_c23_temporal_vit** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.3708, BalAcc=0.5442, MCC=0.0927

### Ranking by Balanced Accuracy (if available)

1. **ffpp_c23_temporal_vit** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.3708, BalAcc=0.5442, MCC=0.0927
2. **fakeavceleb_wav_encoder_baseline** (FakeAVCeleb) — Type: Audio, Backbone: WAV encoder, Acc=0.9480, BalAcc=0.5000, MCC=0.0000
3. **ffpp_c23_temporal_mobilenetv2** (FaceForensics++ C23) — Type: Visual, Backbone: MobileNetV2, Acc=0.5674, BalAcc=0.4348, MCC=-0.1210

### Ranking by MCC (if available)

1. **ffpp_c23_temporal_vit** (FaceForensics++ C23) — Type: Visual, Backbone: ViT, Acc=0.3708, BalAcc=0.5442, MCC=0.0927
2. **fakeavceleb_wav_encoder_baseline** (FakeAVCeleb) — Type: Audio, Backbone: WAV encoder, Acc=0.9480, BalAcc=0.5000, MCC=0.0000
3. **ffpp_c23_temporal_mobilenetv2** (FaceForensics++ C23) — Type: Visual, Backbone: MobileNetV2, Acc=0.5674, BalAcc=0.4348, MCC=-0.1210

## Full comparison table

| Model                             | Dataset             | Type      | Backbone    | Acc    | Bal Acc | F1(w)  | F1(m)  | AUC    | AP     | MCC     |
|-----------------------------------|---------------------|-----------|-------------|--------|---------|--------|--------|--------|--------|---------|
| fakeavceleb_audio_resnet_baseline | FakeAVCeleb         | Audio     | ResNet18    | 0.9672 | N/A     | 0.9603 | 0.7648 | N/A    | N/A    | N/A     |
| fakeavceleb_av_fusion_v1          | FakeAVCeleb         | AV fusion | Unknown     | 0.9990 | N/A     | 0.9990 | 0.9952 | N/A    | N/A    | N/A     |
| fakeavceleb_wav_encoder_baseline  | FakeAVCeleb         | Audio     | WAV encoder | 0.9480 | 0.5000  | 0.9226 | 0.4866 | N/A    | N/A    | 0.0000  |
| ffpp_c23_mobilenet_baseline       | FaceForensics++ C23 | Visual    | MobileNetV2 | 0.4087 | N/A     | 0.6776 | 0.4898 | 0.4496 | 0.7402 | N/A     |
| ffpp_c23_temporal_mobilenetv2     | FaceForensics++ C23 | Visual    | MobileNetV2 | 0.5674 | 0.4348  | 0.5833 | 0.4377 | N/A    | N/A    | -0.1210 |
| ffpp_c23_temporal_vit             | FaceForensics++ C23 | Visual    | ViT         | 0.3708 | 0.5442  | 0.3607 | 0.3703 | N/A    | N/A    | 0.0927  |
| ffpp_c23_vit_baseline             | FaceForensics++ C23 | Visual    | ViT         | 0.6871 | N/A     | 0.6578 | 0.4820 | N/A    | N/A    | N/A     |
| ffpp_c23_xception_baseline        | FaceForensics++ C23 | Visual    | Xception    | 0.7458 | N/A     | 0.7162 | 0.5650 | N/A    | N/A    | N/A     |

## Observations

- Fusion models outperform single-modality models on FakeAVCeleb.
- Xception is the strongest visual backbone on FaceForensics++ C23.
- Some models show high accuracy but weak MCC, suggesting class imbalance effects.
- Temporal variants do not consistently outperform static frame models.