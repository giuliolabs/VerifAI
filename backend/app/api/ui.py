from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
def home_page():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>VerifAI - Deepfake Detection</title>
  <style>
    :root{
      --bg:#070A12;
      --card:#0C1224;
      --card2:#0B1020;
      --text:#E8EEFF;
      --muted:#9AA6C7;
      --accent:#7C3AED;
      --accent2:#22D3EE;
      --danger:#FF4D6D;
      --ok:#2EE59D;
      --border:rgba(255,255,255,.10);
      --shadow: 0 20px 60px rgba(0,0,0,.55);
      --radius: 18px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji";
      background:
        radial-gradient(800px 400px at 10% 10%, rgba(124,58,237,.30), transparent 60%),
        radial-gradient(700px 420px at 90% 20%, rgba(34,211,238,.22), transparent 55%),
        radial-gradient(800px 500px at 50% 90%, rgba(255,77,109,.10), transparent 60%),
        var(--bg);
      color:var(--text);
      min-height:100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:28px 16px;
    }
    .wrap{width:min(980px, 100%);}
    .header{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:16px;
      margin-bottom:16px;
    }
    .brand{
      display:flex;
      gap:14px;
      align-items:center;
    }
    .brandText {
      display:flex;
      flex-direction:column;
    }
    .logo{
      width:90px;height:90px;border-radius:14px;
      background: linear-gradient(135deg, rgba(124,58,237,1), rgba(34,211,238,1));
      box-shadow: 0 10px 30px rgba(124,58,237,.25);
      position:relative;
      overflow:hidden;
    }
    .logo:after{
      content:"";
      position:absolute; inset:-60% -40%;
      background: repeating-linear-gradient(90deg, rgba(255,255,255,.18) 0 2px, transparent 2px 10px);
      transform: rotate(20deg);
      opacity:.35;
    }
    h1{margin:0;font-size:28px;letter-spacing:.2px}
    .sub{margin:6px 0 0;color:var(--muted);font-size:14px;line-height:1.4}
    .pill{
      border:1px solid var(--border);
      border-radius:999px;
      padding:8px 12px;
      color:var(--muted);
      font-size:12px;
      background: rgba(255,255,255,.03);
      align-self:flex-start;
    }
    .grid{
      display:grid;
      grid-template-columns: 1.2fr .8fr;
      gap:16px;
    }
    @media (max-width: 860px){
      .grid{grid-template-columns:1fr}
    }
    .card{
      background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
      border:1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow:hidden;
    }
    .card .top{
      padding:18px 18px 14px;
      border-bottom:1px solid var(--border);
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      background: rgba(0,0,0,.20);
    }
    .card .top b{font-size:14px}
    .card .top span{font-size:12px;color:var(--muted)}
    .card .body{padding:18px}
    .drop{
      border:1px dashed rgba(255,255,255,.18);
      border-radius:16px;
      padding:18px;
      background: rgba(0,0,0,.22);
      display:flex;
      gap:14px;
      align-items:center;
      justify-content:space-between;
      flex-wrap:wrap;
    }
    .drop .left{
      display:flex; gap:12px; align-items:center;
      min-width: 260px;
    }
    .chip{
      width:38px;height:38px;border-radius:14px;
      border:1px solid rgba(255,255,255,.14);
      background: radial-gradient(circle at 30% 30%, rgba(34,211,238,.25), transparent 60%),
                  radial-gradient(circle at 70% 70%, rgba(124,58,237,.30), transparent 60%),
                  rgba(255,255,255,.03);
      display:grid; place-items:center;
      font-weight:700;
    }
    .meta .t{font-size:13px}
    .meta .m{font-size:12px;color:var(--muted);margin-top:2px}
    input[type=file]{display:none}
    .btn{
      border:0;
      border-radius: 14px;
      padding: 11px 14px;
      font-weight: 650;
      cursor:pointer;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      color:#081022;
      box-shadow: 0 12px 28px rgba(124,58,237,.22);
      transition: transform .08s ease, filter .2s ease;
      user-select:none;
    }
    .btn:hover{filter:brightness(1.05)}
    .btn:active{transform:translateY(1px)}
    .btn.secondary{
      background: rgba(255,255,255,.06);
      color: var(--text);
      border:1px solid rgba(255,255,255,.12);
      box-shadow:none;
      font-weight:600;
    }
    .row{display:flex; gap:10px; align-items:center; flex-wrap:wrap}
    .hint{color:var(--muted);font-size:12px;margin-top:10px}
    .bar{
      margin-top:14px;
      height:10px;
      width:100%;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
      overflow:hidden;
      border:1px solid rgba(255,255,255,.10);
    }
    .bar > div{
      height:100%;
      width:0%;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      transition: width .35s ease;
    }
    .result{
      margin-top:16px;
      padding:14px;
      border-radius:16px;
      border:1px solid rgba(255,255,255,.10);
      background: rgba(0,0,0,.22);
    }
    .result .k{display:flex; justify-content:space-between; align-items:center; gap:12px}
    .badge{
      padding:6px 10px;
      border-radius:999px;
      font-size:12px;
      border:1px solid rgba(255,255,255,.14);
      background: rgba(255,255,255,.04);
      color: var(--muted);
      white-space:nowrap;
    }
    .badge.ok{color: var(--ok); border-color: rgba(46,229,157,.35); background: rgba(46,229,157,.08)}
    .badge.bad{color: var(--danger); border-color: rgba(255,77,109,.35); background: rgba(255,77,109,.08)}
    pre{
      margin:10px 0 0;
      padding:12px;
      border-radius:14px;
      background: #060914;
      border:1px solid rgba(255,255,255,.08);
      overflow:auto;
      font-size:12px;
      color:#D6DEFF;
      max-height: 260px;
    }
    .side p{margin:0;color:var(--muted);font-size:13px;line-height:1.5}
    .list{margin-top:12px;display:grid;gap:10px}
    .item{
      padding:12px;
      border-radius:16px;
      border:1px solid rgba(255,255,255,.10);
      background: rgba(0,0,0,.18);
    }
    .item b{display:block;font-size:13px;margin-bottom:4px}
    .item span{color:var(--muted);font-size:12px}
    .tiny{font-size:12px;color:var(--muted);margin-top:10px}
    a{color:inherit}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="brand">
        <img src="/static/verifai-logo.png" alt="VerifAI Logo" style="height:90px;">
        <div>
          <h1>VerifAI – Deepfake Detection</h1>
          <div class="sub">Upload a video to get a fake probability.</div>
        </div>
      </div>
      <div class="pill">FastAPI</div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="top">
          <b>Try it</b>
          <span id="fileInfo">No file selected</span>
        </div>
        <div class="body">
          <div class="drop">
            <div class="left">
              <div class="chip">AI</div>
              <div class="meta">
                <div class="t">Choose a video file</div>
                <div class="m">Supported: .mp4 .mov .avi Recommended: &lt; 25MB</div>
              </div>
            </div>

            <div class="row">
              <label class="btn secondary" for="file">Select file</label>
              <button class="btn" id="detectBtn" onclick="run()" disabled>Detect</button>
            </div>

            <input id="file" type="file" accept=".mp4,.mov,.avi"/>
          </div>

          <div class="bar" aria-label="progress">
            <div id="barFill"></div>
          </div>

          <div class="hint" id="status">Waiting for input…</div>

          <div class="result">
            <div class="k">
              <div>
                <div style="font-weight:700">Result</div>
                <div class="tiny">If you see HTTP 502, the server crashed or timed out during inference.</div>
              </div>
              <div id="badge" class="badge">Idle</div>
            </div>
            <pre id="out">{}</pre>
          </div>
        </div>
      </div>

      <div class="card side">
        <div class="top">
          <b>What this demo does</b>
          <span>Quick notes</span>
        </div>
        <div class="body">
          <p>
            This UI sends your selected video to the API endpoint and shows the JSON response.
            For best stability, use short clips.
          </p>
          <div class="list">
            <div class="item">
              <b>Endpoint</b>
              <span><code>POST /api/predict</code> (multipart upload)</span>
            </div>
            <div class="item">
              <b>Expected response</b>
              <span><code>{"label":"fake","prob_fake":"83.92%","mode": "hybrid_av"}</code></span>
            </div>
            <div class="item">
              <b>Docs</b>
              <span>Swagger UI at <code>/docs</code></span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
const fileEl = document.getElementById("file");
const fileInfo = document.getElementById("fileInfo");
const detectBtn = document.getElementById("detectBtn");
const statusEl = document.getElementById("status");
const outEl = document.getElementById("out");
const badgeEl = document.getElementById("badge");
const barFill = document.getElementById("barFill");

// TODO: change this to your real key (or set it to "" to disable auth on backend)
const API_KEY = "verifai_2026_v1.0_key_2102";

function setBadge(kind, text){
  badgeEl.className = "badge" + (kind ? (" " + kind) : "");
  badgeEl.textContent = text;
}

function setProgress(pct){
  barFill.style.width = pct + "%";
}

fileEl.addEventListener("change", () => {
  if(!fileEl.files.length){
    fileInfo.textContent = "No file selected";
    detectBtn.disabled = true;
    return;
  }
  const f = fileEl.files[0];
  fileInfo.textContent = `${f.name} • ${(f.size/1024/1024).toFixed(1)} MB`;
  detectBtn.disabled = false;
  statusEl.textContent = "Ready.";
  setBadge("", "Ready");
  setProgress(0);
  outEl.textContent = "{}";
});

async function run(){
  const f = fileEl.files[0];
  if(!f) return;

  // Client-side size guard (prevents obvious free-tier failures)
  const maxMB = 25;
  if(f.size > maxMB * 1024 * 1024){
    setBadge("bad", "Too large");
    statusEl.textContent = `File is ${(f.size/1024/1024).toFixed(1)}MB. Please upload under ${maxMB}MB.`;
    return;
  }

  setBadge("", "Running…");
  statusEl.textContent = "Uploading + running inference…";
  outEl.textContent = "";
  setProgress(20);

  const form = new FormData();
  form.append("file", f);

  try{
    const headers = {};
    if (API_KEY) headers["X-API-Key"] = API_KEY;

    const res = await fetch("/api/predict", {
      method: "POST",
      headers,
      body: form
    });

    setProgress(75);
    const text = await res.text();
    outEl.textContent = text || "{}";

    if(res.ok){
      setBadge("ok", "OK");
      statusEl.textContent = "Done";
      setProgress(100);
    }else{
      setBadge("bad", "Error");
      statusEl.textContent = `Error (HTTP ${res.status})`;
      setProgress(100);
    }
  }catch(e){
    setBadge("bad", "Network");
    statusEl.textContent = "Request failed";
    outEl.textContent = String(e);
    setProgress(100);
  }
}
</script>
</body>
</html>
"""
