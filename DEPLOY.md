# Deploying VerifAI (£0/month)

VerifAI runs **entirely in the visitor's browser**: the trained checkpoints are exported to ONNX
(int8 weight compression, ~32 MB) and executed with ONNX Runtime Web (WebAssembly). No server,
no API key, no cold starts, and videos never leave the user's device.

| Piece | Where | Cost |
|---|---|---|
| Website + models (`website/`) | GitHub Pages, branch `gh-pages` → https://vrifai.com | Free |
| Python API (`backend/`) | Optional — run locally with `uvicorn backend.main:app` | Free |

## Update the live site

```bash
python scripts/export_onnx.py                 # only if checkpoints changed
git add -A && git commit -m "Update site"
git push origin main
git push origin "$(git subtree split --prefix website)":refs/heads/gh-pages
```

## One-time GitHub Pages + domain setup

1. GitHub → **Settings → Pages → Source: Deploy from a branch → `gh-pages` / `(root)`**.
2. **Custom domain:** `vrifai.com`, then tick **Enforce HTTPS** once DNS is verified.
3. At the domain registrar, remove parking records and add:

| Type | Host | Value |
|---|---|---|
| A | @ | 185.199.108.153 |
| A | @ | 185.199.109.153 |
| A | @ | 185.199.110.153 |
| A | @ | 185.199.111.153 |
| CNAME | www | giuliolabs.github.io |

## Parity with the Python pipeline

- `scripts/check_onnx_parity.py clip.mp4 …` — PyTorch vs ONNX on identical tensors (labels 10/10, max Δ ≈ 6 pts on synthetic clips).
- The browser reproduces the backend preprocessing: MFCC matches librosa to ~1e-6; the audio path
  (ffmpeg 44.1 kHz resampling + moviepy 16 kHz sample picking) is replicated exactly.
- Frames: browsers' video decoders differ from OpenCV by ~1 intensity level, which can move scores
  by a few points on borderline clips.

## Notes

- Browsers cannot decode AVI; use MP4 (H.264/AAC), MOV or WebM.
- `deploy/hf-space/` (Docker Space) needs a paid Hugging Face PRO plan and is kept only as an option.
