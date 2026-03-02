# Model Comparison Summary

Auto-generated from `experiments/results/*`.

| Model                             | Dataset             | Type      | Backbone    | Acc    | Bal Acc | F1(w)  | F1(m)  | AUC    | AP     | MCC     |
|-----------------------------------|---------------------|-----------|-------------|--------|---------|--------|--------|--------|--------|---------|
| fakeavceleb_audio_resnet_baseline | FakeAVCeleb         | AV fusion | ResNet18    | 0.9672 | N/A     | 0.9603 | 0.7648 | N/A    | N/A    | N/A     |
| fakeavceleb_av_fusion_v1          | FakeAVCeleb         | AV fusion | Unknown     | 0.9990 | N/A     | 0.9990 | 0.9952 | N/A    | N/A    | N/A     |
| fakeavceleb_wav_encoder_baseline  | FakeAVCeleb         | AV fusion | Unknown     | 0.9480 | 0.5000  | 0.9226 | 0.4866 | N/A    | N/A    | 0.0000  |
| ffpp_c23_mobilenet_baseline       | FaceForensics++ C23 | AV fusion | MobileNetV2 | 0.4087 | N/A     | 0.6776 | 0.4898 | 0.4496 | 0.7402 | N/A     |
| ffpp_c23_temporal_mobilenetv2     | FaceForensics++ C23 | AV fusion | MobileNetV2 | 0.5674 | 0.4348  | 0.5833 | 0.4377 | N/A    | N/A    | -0.1210 |
| ffpp_c23_temporal_vit             | FaceForensics++ C23 | AV fusion | ViT         | 0.3708 | 0.5442  | 0.3607 | 0.3703 | N/A    | N/A    | 0.0927  |
| ffpp_c23_vit_baseline             | FaceForensics++ C23 | AV fusion | ViT         | 0.6871 | N/A     | 0.6578 | 0.4820 | N/A    | N/A    | N/A     |
| ffpp_c23_xception_baseline        | FaceForensics++ C23 | AV fusion | Xception    | 0.7458 | N/A     | 0.7162 | 0.5650 | N/A    | N/A    | N/A     |
| preprocessing_checks              | DeeperForensics     | Audio     | Unknown     | N/A    | N/A     | N/A    | N/A    | N/A    | N/A    | N/A     |