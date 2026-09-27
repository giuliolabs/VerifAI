# Deploying VerifAI for £0/month

| Piece | Where it runs | Cost |
|---|---|---|
| Website (`website/`) | GitHub Pages → **https://vrifai.com** | Free |
| Model API (`backend/` + checkpoints) | Hugging Face Spaces, Docker, CPU basic (2 vCPU / 16 GB RAM) | Free |
| Keep-alive ping (`.github/workflows/keepalive.yml`) | GitHub Actions, every 12 h | Free |

The only thing you pay for is the domain itself.

---

## 1. Put the API on Hugging Face (≈10 min)

1. Create a free account at <https://huggingface.co/join> (ideally username `giuliolabs`).
2. Create a **Write** token: <https://huggingface.co/settings/tokens>.
3. From the repo root on your computer:

   ```bash
   pip install -U huggingface_hub
   huggingface-cli login                      # paste the token
   python deploy/push_to_hf_space.py --space giuliolabs/verifai
   ```

   This uploads only what the server needs (~131 MB: `backend/`, `ml/`, the 3 final checkpoints + fusion model) plus the Space `Dockerfile`.
4. Open <https://huggingface.co/spaces/giuliolabs/verifai> and wait for **Running** (first build ≈ 5–10 min).
5. Test: <https://giuliolabs-verifai.hf.space/api/health> → `{"status":"ok",...}`

> **Different username?** The API URL is `https://<username>-verifai.hf.space`. Put it in `website/config.js` (`API_BASE`) and in the README.

Do **not** set `VERIFAI_API_KEY` on the Space — the public website can't keep a key secret. Abuse protection comes from the 25 MB limit, file validation and CORS (only vrifai.com may call the API from a browser). If the domain changes, set the Space variable `VERIFAI_ALLOWED_ORIGINS` (comma-separated).

## 2. Publish the website with GitHub Pages (≈5 min)

1. Copy these files into your local clone, then commit and push to `main`:
   `website/`, `.github/workflows/`, `deploy/`, `backend/main.py`, `DEPLOY.md`, `README.md`, `LICENSE.md`, `requirements.txt`, `.gitignore`
2. GitHub → **giuliolabs/VerifAI → Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. **Actions** tab → *Deploy website* should run green (re-run it once if it ran before step 2).
4. **Settings → Pages → Custom domain:** `vrifai.com` → Save.

## 3. Point vrifai.com at GitHub (at your domain registrar)

Delete the registrar's parking/placeholder records (any existing `A`, `ALIAS` or "URL redirect" on `@` and `www`), then add:

| Type | Host | Value |
|---|---|---|
| A | @ | 185.199.108.153 |
| A | @ | 185.199.109.153 |
| A | @ | 185.199.110.153 |
| A | @ | 185.199.111.153 |
| AAAA *(optional)* | @ | 2606:50c0:8000::153 |
| AAAA *(optional)* | @ | 2606:50c0:8001::153 |
| AAAA *(optional)* | @ | 2606:50c0:8002::153 |
| AAAA *(optional)* | @ | 2606:50c0:8003::153 |
| CNAME | www | giuliolabs.github.io |

DNS takes minutes to a few hours. When GitHub's Pages settings show the DNS check as successful, tick **Enforce HTTPS**.

Recommended: verify the domain under your GitHub **Settings → Pages → Add a verified domain** (prevents someone else claiming it).

## 4. Finish

- Open <https://vrifai.com>, upload a short clip, confirm you get a verdict.
- Cancel the Render service/plan.
- *Actions → Keep API awake → Run workflow* once to check the ping works.

## Local development

```bash
uvicorn backend.main:app --reload            # API on :8000
cd website && python -m http.server 5500     # site on :5500
```
For local testing, temporarily set `API_BASE: "http://127.0.0.1:8000"` in `website/config.js` (localhost:5500 is already allowed by CORS).

## Notes

- Free Spaces sleep after ~48 h without traffic; the keep-alive workflow prevents that. If it does sleep, the site shows *Waking server…* and retries automatically (~30–60 s).
- GitHub disables scheduled workflows in repos with no activity for 60 days — re-enable from the Actions tab if needed.
- Typical inference time on the free CPU is a few seconds per clip.
