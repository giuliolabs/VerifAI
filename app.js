/* VerifAI website — UI for the in-browser detection engine
 * © 2026 Giulio Labs
 */
(() => {
  "use strict";

  const CFG = Object.assign(
    { MODEL_BASE: "models/", GITHUB_URL: "https://github.com/giuliolabs/VerifAI", MAX_FILE_MB: 100 },
    window.VERIFAI_CONFIG || {}
  );
  const ALLOWED = [".mp4", ".mov", ".m4v", ".webm"];
  const N_FRAMES = 5;
  const MODE_LABEL = { hybrid_av: "Audio + visual fusion", visual_only_fallback: "Visual only (no usable audio)" };

  const $ = (id) => document.getElementById(id);
  const el = {
    status: $("apiStatus"), drop: $("drop"), input: $("fileInput"), work: $("work"),
    preview: $("preview"), fileName: $("fileName"), fileSize: $("fileSize"), frames: $("frames"),
    actions: $("actions"), consent: $("consent"), analyze: $("analyzeBtn"), reset: $("resetBtn"),
    progress: $("progress"), bar: $("barFill"), steps: $("steps"), cold: $("coldNote"), loadNote: $("loadNote"),
    result: $("result"), gauge: $("gaugeFill"), tick: $("thresholdTick"), probNum: $("probNum"),
    verdict: $("verdictText"), verdictSub: $("verdictSub"), mode: $("modeChip"), thresh: $("threshChip"),
    time: $("timeChip"), again: $("againBtn"), json: $("jsonBtn"), error: $("error"),
  };

  let file = null, objectUrl = null, lastJson = null, busy = false, modelsReady = false;

  document.querySelectorAll('a[href^="https://github.com/giuliolabs/VerifAI"]').forEach((a) => {
    a.href = a.href.replace("https://github.com/giuliolabs/VerifAI", CFG.GITHUB_URL);
  });

  const engine = new window.VerifaiEngine({
    modelBase: CFG.MODEL_BASE,
    onProgress: (loaded, total) => {
      const pct = total ? Math.min(100, Math.round((loaded / total) * 100)) : 0;
      const txt = `${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(0)} MB`;
      el.loadNote.textContent = `· ${txt}`;
      if (!modelsReady) setStatus("loading", `Downloading model ${pct}%`);
      if (busy) el.bar.style.width = `${Math.round(pct * 0.35)}%`;
    },
    onStatus: (t) => { if (!modelsReady) setStatus("loading", t); },
  });

  function setStatus(state, text) {
    el.status.dataset.state = state;
    el.status.querySelector("span").textContent = text;
  }

  function capabilityCheck() {
    const ok = typeof WebAssembly === "object" && "Worker" in window && window.OfflineAudioContext;
    if (!ok) setStatus("offline", "Browser not supported");
    return ok;
  }

  // Warm the model download in the background once a file is chosen (or on idle for returning visitors)
  function preload() {
    if (!capabilityCheck()) return;
    engine.load().then(() => { modelsReady = true; setStatus("online", "Model ready · on-device"); el.loadNote.textContent = ""; })
      .catch((e) => { setStatus("offline", "Model failed to load"); console.error(e); });
  }
  if ("caches" in window) {
    caches.has("verifai-models-v1").then((has) => { if (has) preload(); }).catch(() => {});
  }

  /* ---------------- file selection ---------------- */
  el.input.addEventListener("change", () => el.input.files[0] && pick(el.input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
  el.drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && pick(e.dataTransfer.files[0]));
  $("detector").addEventListener("dragover", (e) => e.preventDefault());
  $("detector").addEventListener("drop", (e) => { e.preventDefault(); if (!busy && e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); });

  function pick(f) {
    const ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    el.preview.parentElement.hidden = false; el.frames.parentElement.hidden = false;
    resetView();
    if (ext === ".avi") return showError("AVI files can't be decoded by web browsers. Convert the clip to MP4 (H.264) and try again.", true);
    if (!ALLOWED.includes(ext)) return showError(`Unsupported file type (${ext || "unknown"}). Please use MP4, MOV or WebM.`, true);
    if (f.size > CFG.MAX_FILE_MB * 1024 * 1024) return showError(`That file is ${fmtSize(f.size)} — the limit is ${CFG.MAX_FILE_MB} MB. Trim the clip and try again.`, true);

    file = f;
    el.drop.hidden = true; el.work.hidden = false;
    el.fileName.textContent = f.name; el.fileSize.textContent = fmtSize(f.size);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(f);
    el.preview.src = objectUrl;
    renderFramePlaceholders();
    const token = objectUrl;
    engine.extractFrames(objectUrl, (i, c) => { if (token === objectUrl) placeFrame(i, c); })
      .catch(() => {
        if (token !== objectUrl) return;
        el.frames.querySelectorAll(".ph").forEach((p) => (p.style.animation = "none"));
        setFramesLabel("This browser can't decode this video — try an MP4 (H.264) or WebM file");
      });
    setFramesLabel("What the model sees · 5 frames @ 224×224");
    updateAnalyzeBtn();
    if (!modelsReady) preload();
  }

  const fmtSize = (b) => (b > 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);
  const setFramesLabel = (t) => (el.frames.parentElement.querySelector(".frames-label span").textContent = t);

  function renderFramePlaceholders() {
    el.frames.querySelectorAll("canvas,.ph").forEach((n) => n.remove());
    for (let i = 0; i < N_FRAMES; i++) {
      const ph = document.createElement("div"); ph.className = "ph"; ph.dataset.i = i;
      el.frames.appendChild(ph);
    }
  }
  function placeFrame(i, canvas) {
    const slot = el.frames.querySelector(`[data-i="${i}"]`);
    if (slot) { canvas.dataset.i = i; slot.replaceWith(canvas); }
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

  const STAGE_PCT = { load: 2, frames: 38, mfcc: 48, visual: 58, audio: 88, fuse: 96 };
  function setStep(name) {
    let passed = true;
    el.steps.querySelectorAll("li").forEach((li) => {
      if (li.dataset.step === name) { li.className = "active"; passed = false; }
      else li.className = passed ? "done" : "";
    });
    if (name !== "load" || modelsReady) el.bar.style.width = `${STAGE_PCT[name] || 0}%`;
  }

  async function analyse() {
    if (!file || busy) return;
    busy = true; updateAnalyzeBtn();
    el.error.hidden = true; el.actions.hidden = true; el.progress.hidden = false;
    el.cold.hidden = modelsReady;
    el.bar.style.width = "0%";
    const t0 = performance.now();
    try {
      const r = await engine.analyse(file, {
        onStage: (s) => {
          setStep(s);
          if (s === "frames") { modelsReady = true; setStatus("online", "Model ready · on-device"); el.cold.hidden = true; }
          if (s === "visual") el.frames.classList.add("scanning");
        },
        onFrame: (i, c) => placeFrame(i, c),
      });
      el.bar.style.width = "100%";
      el.steps.querySelectorAll("li").forEach((li) => (li.className = "done"));
      await new Promise((res) => setTimeout(res, 200));
      showResult(r, (performance.now() - t0) / 1000);
    } catch (err) {
      console.error(err);
      el.progress.hidden = true; el.actions.hidden = false;
      showError(err.message || "Something went wrong. Please try again.");
    } finally {
      el.frames.classList.remove("scanning");
      busy = false; updateAnalyzeBtn();
    }
  }

  function showResult(r, secs) {
    const p = r.prob, fake = r.label === "fake", thr = r.threshold;
    lastJson = {
      label: r.label, prob_fake: `${(p * 100).toFixed(2)}%`, mode: r.mode,
      ...(r.mode === "hybrid_av" ? { p_visual: +r.pVisual.toFixed(4), p_audio: +r.pAudio.toFixed(4) } : {}),
      threshold: thr, runtime: "onnxruntime-web (wasm)", file: file ? file.name : undefined,
    };

    el.progress.hidden = true; el.result.hidden = false;
    el.gauge.style.stroke = fake ? "var(--fake)" : "var(--real)";
    el.tick.setAttribute("transform", `rotate(${thr * 360} 60 60)`);
    requestAnimationFrame(() => (el.gauge.style.strokeDashoffset = 314.16 * (1 - p)));
    countUp(el.probNum, p * 100);

    el.verdict.textContent = fake ? "Likely manipulated" : "Likely authentic";
    el.verdict.className = fake ? "fake" : "real";
    const gap = Math.abs(p - thr);
    const strength = gap > 0.3 ? "strong" : gap > 0.12 ? "moderate" : "weak";
    el.verdictSub.textContent = fake
      ? `The model found ${strength} evidence of manipulation.`
      : `No ${strength === "strong" ? "" : "clear "}signs of manipulation were found.`;
    if (strength === "weak") el.verdictSub.textContent += " The score is close to the threshold — treat with caution.";
    el.mode.textContent = MODE_LABEL[r.mode] || r.mode;
    el.thresh.textContent = `threshold ${Math.round(thr * 100)}%`;
    el.time.textContent = `${secs.toFixed(1)} s on this device`;
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

  // expose for automated parity tests
  window.__verifai = { engine };
})();
