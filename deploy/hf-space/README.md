---
title: VerifAI API
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
license: other
short_description: Multimodal (audio + visual) deepfake detection API
---

# VerifAI — Deepfake Detection API

Backend for **[vrifai.com](https://vrifai.com)**. Source code: **[github.com/giuliolabs/VerifAI](https://github.com/giuliolabs/VerifAI)**.

| Endpoint | Description |
|---|---|
| `GET /api/health` | Health check |
| `POST /api/predict` | Upload a video (`file`, .mp4/.mov/.avi, ≤ 25 MB) → `{label, prob_fake, mode}` |
| `GET /docs` | Interactive Swagger docs |

```bash
curl -F "file=@clip.mp4" https://<your-space>.hf.space/api/predict
```

© 2026 Giulio Labs — research prototype, not for forensic or legal use.
