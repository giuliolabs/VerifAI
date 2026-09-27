/* VerifAI website — front-end for the VerifAI FastAPI backend
 * © 2026 Giulio Labs
 */
(() => {
  "use strict";

  const CFG = Object.assign(
    { API_BASE: "", GITHUB_URL: "https://github.com/giuliolabs/VerifAI", MAX_UPLOAD_MB: 25 },
    window.VERIFAI_CONFIG || {}
  );
  const API = CFG.API_BASE.replace(/\/+$/, "");
  const ALLOWED = [".mp4", ".mov", ".avi"];
  const N_FRAMES = 5;
  // Decision thresholds used by the backend (see backend/app/services/inference_service.py)
  const THRESHOLDS = { hybrid_av: 0.25, visual_only_fallback: 0.5 };
  const MODE_LABEL = { hybrid_av: "Audio + visual fusion", visual_only_fallback: "Visual only (no usable audio)" };

  const $ = (id) => document.getElementById(id);
  const el = {
    status: $("apiStatus"), drop: $("drop"), input: $("fileInput"), work: $("work"),
    preview: $("preview"), fileName: $("fileName"), fileSize: $("fileSize"), frames: $("frames"),
    actions: $("actions"), consent: $("consent"), analyze: $("analyzeBtn"), reset: $("resetBtn"),
    progress: $("progress"), bar: $("barFill"), steps: $("steps"), cold: $("coldNote"),
    result: $("result"), gauge: $("gaugeFill"), tick: $("thresholdTick"), probNum: $("probNum"),
    verdict: $("verdictText"), verdictSub: $("verdictSub"), mode: $("modeChip"), thresh: $("threshChip"),
    time: $("timeChip"), again: $("againBtn"), json: $("jsonBtn"), error: $("error"),
  };

  let file = null, objectUrl = null, lastJson = null, apiOnline = false, busy = false;

  /* ---------------- links ---------------- */
  document.querySelectorAll('a[href^="https://github.com/giuliolabs/VerifAI"]').forEach((a) => (a.href = CFG.GITHUB_URL));
  const docs = $("apiDocs");
  if (docs) docs.href = API ? `${API}/docs` : CFG.GITHUB_URL;

  /* ---------------- server status ---------------- */
  function setStatus(state, text) {
    el.status.dataset.state = state;
    el.status.querySelector("span").textContent = text;
  }

  async function ping(timeoutMs = 8000) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const r = await fetch(`${API}/api/health`, { signal: ctrl.signal, cache: "no-store" });
      if (!r.ok) return false;
      const j = await r.json().catch(() => ({}));
      return j.status === "ok";
    } catch { return false; } finally { clearTimeout(t); }
  }

  async function watchServer() {
    if (!API) { setStatus("offline", "API not configured"); return; }
    setStatus("checking", "Checking server…");
    const started = Date.now();
    while (Date.now() - started < 4 * 60 * 1000) {
      if (await ping()) { apiOnline = true; setStatus("online", "Model online"); return; }
      setStatus("waking", "Waking server…");
      await sleep(5000);
    }
    setStatus("offline", "Server unavailable");
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /* ---------------- file selection ---------------- */
  el.input.addEventListener("change", () => el.input.files[0] && pick(el.input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
  el.drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && pick(e.dataTransfer.files[0]));
  // allow dropping anywhere on the detector while a file is loaded
  $("detector").addEventListener("dragover", (e) => e.preventDefault());
  $("detector").addEventListener("drop", (e) => { e.preventDefault(); if (!busy && e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); });

  function pick(f) {
    const ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    el.preview.parentElement.hidden = false; el.frames.parentElement.hidden = false;
    resetView();
    if (!ALLOWED.includes(ext)) return showError(`Unsupported file type (${ext || "unknown"}). Please use MP4, MOV or AVI.`, true);
    if (f.size > CFG.MAX_UPLOAD_MB * 1024 * 1024) return showError(`That file is ${fmtSize(f.size)} — the limit is ${CFG.MAX_UPLOAD_MB} MB. Trim the clip and try again.`, true);

    file = f;
    el.drop.hidden = true;
    el.work.hidden = false;
    el.fileName.textContent = f.name;
    el.fileSize.textContent = fmtSize(f.size);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(f);
    el.preview.src = objectUrl;
    renderFramePlaceholders();
    const token = objectUrl;
    extractFrames(objectUrl).catch(() => {
      if (token !== objectUrl) return;
      // Browser can't decode this codec locally — the server still can.
      el.frames.querySelectorAll(".ph").forEach((p) => (p.style.animation = "none"));
      el.frames.parentElement.querySelector(".frames-label span").textContent =
        "Preview unavailable in this browser — the server will still sample 5 frames";
    });
    el.frames.parentElement.querySelector(".frames-label span").textContent = "What the model sees · 5 frames @ 224×224";
    updateAnalyzeBtn();
  }

  function fmtSize(b) { return b > 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`; }

  function renderFramePlaceholders() {
    el.frames.querySelectorAll("canvas,.ph").forEach((n) => n.remove());
    for (let i = 0; i < N_FRAMES; i++) {
      const ph = document.createElement("div"); ph.className = "ph";
      el.frames.appendChild(ph);
    }
  }

  // Mirrors the backend: np.linspace(0, total-1, 5) frame positions, resized to 224x224
  async function extractFrames(url) {
    const v = document.createElement("video");
    v.muted = true; v.playsInline = true; v.preload = "auto"; v.src = url;
    await once(v, "loadeddata", 10000);
    const dur = v.duration;
    if (!isFinite(dur) || dur <= 0) throw new Error("no duration");
    const phs = [...el.frames.querySelectorAll(".ph")];
    for (let i = 0; i < N_FRAMES; i++) {
      const t = Math.min(dur - 0.05, (dur * i) / (N_FRAMES - 1));
      v.currentTime = Math.max(0, t);
      await once(v, "seeked", 5000);
      const c = document.createElement("canvas");
      c.width = 224; c.height = 224;
      c.getContext("2d").drawImage(v, 0, 0, 224, 224);
      c.title = `t = ${t.toFixed(2)} s`;
      phs[i].replaceWith(c);
    }
    v.removeAttribute("src"); v.load();
  }

  function once(target, ev, timeout) {
    return new Promise((res, rej) => {
      const t = setTimeout(() => { target.removeEventListener(ev, h); rej(new Error("timeout")); }, timeout);
      const h = () => { clearTimeout(t); res(); };
      target.addEventListener(ev, h, { once: true });
    });
  }

  el.consent.addEventListener("change", updateAnalyzeBtn);
  function updateAnalyzeBtn() { el.analyze.disabled = !(file && el.consent.checked) || busy; }

  el.reset.addEventListener("click", () => { hardReset(); el.input.click(); });
  el.again.addEventListener("click", () => { hardReset(); el.input.click(); });

  function hardReset() {
    file = null; el.input.value = "";
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    el.preview.removeAttribute("src"); el.preview.load();
    resetView();
    el.work.hidden = true; el.drop.hidden = false;
  }

  function resetView() {
    el.error.hidden = true; el.result.hidden = true; el.progress.hidden = true;
    el.actions.hidden = false; el.frames.classList.remove("scanning");
    el.gauge.style.strokeDashoffset = 314.16;
  }

  /* ---------------- analysis ---------------- */
  el.analyze.addEventListener("click", analyse);

  function setStep(name) {
    let passed = true;
    el.steps.querySelectorAll("li").forEach((li) => {
      if (li.dataset.step === name) { li.className = "active"; passed = false; }
      else li.className = passed ? "done" : "";
    });
  }

  async function analyse() {
    if (!file || busy) return;
    busy = true; updateAnalyzeBtn();
    el.error.hidden = true; el.actions.hidden = true; el.progress.hidden = false;
    el.cold.hidden = apiOnline;
    el.bar.style.width = "0%";
    setStep("upload");
    const t0 = performance.now();

    let stepTimer = null;
    try {
      const data = await upload(file, (p) => {
        el.bar.style.width = `${Math.round(p * 55)}%`;
        if (p >= 1 && !stepTimer) {
          el.frames.classList.add("scanning");
          const seq = ["frames", "audio", "fuse"]; let i = 0; let w = 55;
          setStep(seq[0]);
          stepTimer = setInterval(() => {
            i = Math.min(i + 1, seq.length - 1); setStep(seq[i]);
            w = Math.min(w + 12, 92); el.bar.style.width = `${w}%`;
          }, 1800);
        }
      });
      clearInterval(stepTimer);
      el.bar.style.width = "100%";
      el.steps.querySelectorAll("li").forEach((li) => (li.className = "done"));
      apiOnline = true; setStatus("online", "Model online");
      await sleep(250);
      showResult(data, (performance.now() - t0) / 1000);
    } catch (err) {
      clearInterval(stepTimer);
      el.progress.hidden = true; el.actions.hidden = false;
      showError(err.message || "Something went wrong. Please try again.");
    } finally {
      el.frames.classList.remove("scanning");
      busy = false; updateAnalyzeBtn();
    }
  }

  function upload(f, onProgress, attempt = 0) {
    return new Promise((resolve, reject) => {
      if (!API) return reject(new Error("The API URL is not configured (see config.js)."));
      const fd = new FormData(); fd.append("file", f, f.name);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API}/api/predict`);
      xhr.timeout = 180000;
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.upload.onload = () => onProgress(1);
      xhr.onload = () => {
        let body = null; try { body = JSON.parse(xhr.responseText); } catch {}
        if (xhr.status >= 200 && xhr.status < 300 && body && body.label) return resolve(body);
        // cold start: HF returns 502/503 while the container boots
        if ([502, 503, 504].includes(xhr.status) && attempt < 8) {
          setStatus("waking", "Waking server…"); el.cold.hidden = false;
          return sleep(8000).then(() => upload(f, onProgress, attempt + 1)).then(resolve, reject);
        }
        const detail = body && (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
        reject(new Error(detail || `Server returned ${xhr.status}.`));
      };
      xhr.onerror = () => {
        if (attempt < 8) {
          setStatus("waking", "Waking server…"); el.cold.hidden = false;
          return sleep(8000).then(() => upload(f, onProgress, attempt + 1)).then(resolve, reject);
        }
        setStatus("offline", "Server unavailable");
        reject(new Error("Couldn't reach the VerifAI server. It may be restarting — please try again in a minute."));
      };
      xhr.ontimeout = () => reject(new Error("The analysis timed out. Try a shorter clip."));
      xhr.send(fd);
    });
  }

  function showResult(data, secs) {
    lastJson = data;
    const p = Math.max(0, Math.min(1, parseFloat(String(data.prob_fake).replace("%", "")) / 100 || 0));
    const fake = data.label === "fake";
    const thr = THRESHOLDS[data.mode] ?? 0.5;

    el.progress.hidden = true; el.result.hidden = false;
    el.gauge.style.stroke = fake ? "var(--fake)" : "var(--real)";
    el.tick.setAttribute("transform", `rotate(${thr * 360} 60 60)`);
    requestAnimationFrame(() => (el.gauge.style.strokeDashoffset = 314.16 * (1 - p)));
    countUp(el.probNum, p * 100);

    el.verdict.textContent = fake ? "Likely manipulated" : "Likely authentic";
    el.verdict.className = fake ? "fake" : "real";
    const conf = fake ? p : 1 - p;
    const strength = Math.abs(p - thr) > 0.3 ? "strong" : Math.abs(p - thr) > 0.12 ? "moderate" : "weak";
    el.verdictSub.textContent = fake
      ? `The model found ${strength} evidence of manipulation.`
      : `No ${strength === "strong" ? "" : "clear "}signs of manipulation were found.`;
    if (strength === "weak") el.verdictSub.textContent += " The score is close to the threshold — treat with caution.";
    el.mode.textContent = MODE_LABEL[data.mode] || data.mode;
    el.thresh.textContent = `threshold ${Math.round(thr * 100)}%`;
    el.time.textContent = `${secs.toFixed(1)} s`;
    el.result.dataset.conf = conf.toFixed(3);
    el.actions.hidden = true;
  }

  function countUp(node, target) {
    const start = performance.now(), dur = 1000;
    const step = (now) => {
      const k = Math.min(1, (now - start) / dur), e = 1 - Math.pow(1 - k, 3);
      node.textContent = (target * e).toFixed(target >= 10 ? 0 : 1);
      if (k < 1) requestAnimationFrame(step); else node.textContent = target.toFixed(1);
    };
    requestAnimationFrame(step);
  }

  function showError(msg, standalone) {
    if (standalone) { el.work.hidden = false; el.drop.hidden = false; el.preview.parentElement.hidden = true; el.frames.parentElement.hidden = true; el.actions.hidden = true; }
    else { el.preview.parentElement.hidden = false; el.frames.parentElement.hidden = false; }
    el.error.textContent = msg; el.error.hidden = false;
  }

  /* ---------------- copy helpers ---------------- */
  async function copy(text, btn) {
    try { await navigator.clipboard.writeText(text); } catch {
      const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); } catch {} ta.remove();
    }
    const old = btn.textContent; btn.textContent = "Copied ✓"; setTimeout(() => (btn.textContent = old), 1400);
  }
  el.json.addEventListener("click", () => lastJson && copy(JSON.stringify(lastJson, null, 2), el.json));
  $("copyCite").addEventListener("click", (e) => copy($("bibtex").textContent, e.currentTarget));

  watchServer();
})();
