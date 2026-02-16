from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/try", response_class=HTMLResponse)
def try_page():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>VerifAI – Try Detection</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; }
    button { padding: 10px 14px; border-radius: 10px; border: 0; cursor: pointer; }
    input { margin: 10px 0; }
    pre { background: #111; color: #0f0; padding: 12px; border-radius: 10px; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>VerifAI – Deepfake Detection</h1>
  <div class="card">
    <p>Select a video and click Detect.</p>
    <input id="file" type="file" accept=".mp4,.mov,.avi"/>
    <br/>
    <button onclick="run()">Detect</button>
    <p id="status"></p>
    <pre id="out"></pre>
  </div>

<script>
async function run() {
  const fileInput = document.getElementById("file");
  const status = document.getElementById("status");
  const out = document.getElementById("out");
  out.textContent = "";
  status.textContent = "";

  if (!fileInput.files.length) {
    status.textContent = "Please select a file first.";
    return;
  }

  const f = fileInput.files[0];
  const form = new FormData();
  form.append("file", f);

  status.textContent = "Uploading + running inference...";

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const text = await res.text();
    out.textContent = text;
    status.textContent = res.ok ? "Done" : ("Error (HTTP " + res.status + ")");
  } catch (e) {
    status.textContent = "Request failed";
    out.textContent = String(e);
  }
}
</script>
</body>
</html>
"""
