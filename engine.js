/* VerifAI — in-browser detection engine
 * Mirrors backend/app/services/{preprocess,inference}_service.py:
 *   5 evenly spaced frames -> 224x224 RGB, ImageNet-normalised -> Xception
 *   audio -> 16 kHz mono -> 40 MFCC x 200 frames, standardised -> ResNet-18
 *   usable audio: logistic late fusion (threshold 0.25); otherwise visual only (0.5)
 * © 2026 Giulio Labs
 */
/* global VerifaiMFCC */
(function () {
  "use strict";

  const N_FRAMES = 5, SIZE = 224, SR = 16000;
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  class Engine {
    constructor({ modelBase = "models/", onProgress = () => {}, onStatus = () => {} } = {}) {
      this.base = new URL(modelBase, location.href).href;
      this.files = [
        { key: "visual", name: "visual_xception.onnx", approxBytes: 21.2e6 },
        { key: "audio", name: "audio_resnet18.onnx", approxBytes: 11.2e6 },
      ];
      this.onProgress = onProgress; this.onStatus = onStatus;
      this.worker = null; this.ready = null; this.fusion = null; this.pending = new Map(); this.seq = 0;
      this.stageCb = null;
    }

    _ensureWorker() {
      if (this.worker) return;
      this.worker = new Worker("worker.js");
      this.worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") this.onProgress(m.loaded, m.total);
        else if (m.type === "status") this.onStatus(m.text);
        else if (m.type === "stage") this.stageCb && this.stageCb(m.stage);
        else if (m.type === "loaded") { this._resolveLoad && this._resolveLoad(m); }
        else if (m.type === "result" || m.type === "error") {
          const p = this.pending.get(m.id);
          if (p) { this.pending.delete(m.id); m.type === "error" ? p.reject(new Error(m.message)) : p.resolve(m); }
          else if (m.type === "error" && this._rejectLoad) this._rejectLoad(new Error(m.message));
        }
      };
      this.worker.onerror = (e) => { if (this._rejectLoad) this._rejectLoad(new Error(e.message || "Worker failed to start")); };
    }

    load() {
      if (this.ready) return this.ready;
      this._ensureWorker();
      this.ready = Promise.all([
        fetch(this.base + "fusion.json").then((r) => { if (!r.ok) throw new Error("fusion.json missing"); return r.json(); }),
        new Promise((resolve, reject) => {
          this._resolveLoad = resolve; this._rejectLoad = reject;
          this.worker.postMessage({ type: "load", base: this.base, files: this.files });
        }),
      ]).then(([fusion, info]) => { this.fusion = fusion; return info; })
        .catch((e) => { this.ready = null; throw e; });
      return this.ready;
    }

    /* ---------- video frames ---------- */
    async extractFrames(url, onFrame) {
      const v = document.createElement("video");
      v.muted = true; v.playsInline = true; v.preload = "auto"; v.crossOrigin = "anonymous"; v.src = url;
      await once(v, "loadeddata", 15000, "This browser can't decode the video. Try an MP4 (H.264) file.");
      const dur = v.duration;
      if (!isFinite(dur) || dur <= 0) throw new Error("Couldn't read the video duration.");
      this.lastDuration = dur;
      const canvas = document.createElement("canvas");
      canvas.width = SIZE; canvas.height = SIZE;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "low"; // ~ cv2 INTER_LINEAR
      const out = new Float32Array(N_FRAMES * 3 * SIZE * SIZE);
      const plane = SIZE * SIZE;
      const last = Math.max(0, dur - 0.04); // roughly the last frame
      for (let i = 0; i < N_FRAMES; i++) {
        const t = (last * i) / (N_FRAMES - 1);
        await seek(v, Math.min(t + 0.001, last));
        ctx.drawImage(v, 0, 0, SIZE, SIZE);
        const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
        const off = i * 3 * plane;
        for (let p = 0; p < plane; p++) {
          out[off + p] = (px[p * 4] / 255 - MEAN[0]) / STD[0];
          out[off + plane + p] = (px[p * 4 + 1] / 255 - MEAN[1]) / STD[1];
          out[off + 2 * plane + p] = (px[p * 4 + 2] / 255 - MEAN[2]) / STD[2];
        }
        if (onFrame) {
          const c = document.createElement("canvas"); c.width = SIZE; c.height = SIZE;
          c.getContext("2d").drawImage(canvas, 0, 0); c.title = `t = ${t.toFixed(2)} s`;
          onFrame(i, c);
        }
      }
      v.removeAttribute("src"); v.load();
      return out;
    }

    /* ---------- audio -> 16 kHz mono -> MFCC ---------- */
    /* Reproduces the backend's audio path exactly:
     *   VideoFileClip(...).audio  -> ffmpeg decodes to 44.1 kHz stereo s16 (libswresample)
     *   .write_audiofile(fps=16000) -> moviepy picks the nearest 44.1 kHz sample for each
     *                                  16 kHz timestamp (no low-pass), clips to ±0.99, int16
     *   librosa.load(sr=16000)      -> int16 / 32768, mean of the two channels
     */
    async extractAudio(file, durationSec) {
      if (!window.OfflineAudioContext) return null;
      const bytes = new Uint8Array(await file.arrayBuffer());
      const srcRate = sniffAudioRate(bytes, file.name) || 48000;
      const ctx = new OfflineAudioContext(1, 1, srcRate); // decode at the native rate (no browser resampling)
      let decoded;
      try {
        decoded = await new Promise((res, rej) => {
          const p = ctx.decodeAudioData(bytes.buffer.slice(0), res, rej);
          if (p && p.then) p.then(res, rej);
        });
      } catch { return null; } // no audio track or unsupported codec -> visual-only path
      if (!decoded || decoded.length === 0) return null;

      // 1) ffmpeg "-ac 2": mono is upmixed at -3 dB; >2 channels approximated by averaging
      const ch = decoded.numberOfChannels, n = decoded.length;
      let L, R;
      if (ch === 1) {
        const d = decoded.getChannelData(0); L = new Float32Array(n);
        for (let i = 0; i < n; i++) L[i] = d[i] * Math.SQRT1_2;
        R = L;
      } else if (ch === 2) {
        L = decoded.getChannelData(0); R = decoded.getChannelData(1);
      } else {
        L = new Float32Array(n);
        for (let c = 0; c < ch; c++) { const d = decoded.getChannelData(c); for (let i = 0; i < n; i++) L[i] += d[i] / ch; }
        R = L;
      }
      // 2) ffmpeg "-ar 44100" (libswresample defaults) + s16 conversion
      const r = decoded.sampleRate;
      const toS16 = (x) => { const o = new Int16Array(x.length); for (let i = 0; i < x.length; i++) o[i] = Math.max(-32768, Math.min(32767, roundHalfEven(x[i] * 32768))); return o; };
      const L44 = toS16(r === 44100 ? L : swrResample(L, r, 44100));
      const R44 = R === L ? L44 : toS16(r === 44100 ? R : swrResample(R, r, 44100));

      // 3) moviepy nearest-sample pick at 16 kHz over the clip duration, then int16 quantisation
      const dur = durationSec && isFinite(durationSec) ? Math.round(durationSec * 100) / 100 : L44.length / 44100;
      const K = Math.floor(SR * dur);
      if (K <= 0) return null;
      const y = new Float32Array(K), last = L44.length - 1, step = 1.0 / SR;
      for (let k = 0; k < K; k++) {
        let idx = roundHalfEven(44100 * (step * k));
        if (idx > last) idx = last;
        const a = Math.trunc(32768 * Math.max(-0.99, Math.min(0.99, L44[idx] / 32768)));
        const b = Math.trunc(32768 * Math.max(-0.99, Math.min(0.99, R44[idx] / 32768)));
        y[k] = (Math.fround(a / 32768) + Math.fround(b / 32768)) / 2;
      }
      return y;
    }

    decide(visualLogits, audioLogit) {
      const f = this.fusion;
      if (audioLogit === null || audioLogit === undefined) {
        const p = sigmoid(visualLogits.reduce((a, b) => a + b, 0) / visualLogits.length);
        return { mode: "visual_only_fallback", prob: p, threshold: f.visual_only_threshold, label: p >= f.visual_only_threshold ? "fake" : "real" };
      }
      const pv = visualLogits.map(sigmoid).reduce((a, b) => a + b, 0) / visualLogits.length;
      const pa = sigmoid(audioLogit);
      const p = sigmoid(f.coef[0] * pv + f.coef[1] * pa + f.intercept);
      return { mode: "hybrid_av", prob: p, pVisual: pv, pAudio: pa, threshold: f.hybrid_threshold, label: p >= f.hybrid_threshold ? "fake" : "real" };
    }

    async analyse(file, { onStage = () => {}, onFrame } = {}) {
      const url = URL.createObjectURL(file);
      try {
        onStage("load");
        await this.load();
        onStage("frames");
        const frames = await this.extractFrames(url, onFrame);
        onStage("mfcc");
        const audio = await this.extractAudio(file, this.lastDuration);
        const mfcc = audio ? VerifaiMFCC.modelInput(audio, { sr: SR }) : null;
        onStage("visual");
        this.stageCb = (s) => s === "audio" && onStage("audio");
        const id = ++this.seq;
        const res = await new Promise((resolve, reject) => {
          this.pending.set(id, { resolve, reject });
          const transfer = [frames.buffer]; if (mfcc) transfer.push(mfcc.buffer);
          this.worker.postMessage({ type: "infer", id, frames, mfcc, base: this.base, files: this.files }, transfer);
        });
        onStage("fuse");
        const d = this.decide(res.visualLogits, res.audioLogit);
        return { ...d, visualLogits: res.visualLogits, audioLogit: res.audioLogit, inferenceMs: res.ms };
      } finally { URL.revokeObjectURL(url); }
    }
  }

  function roundHalfEven(v) {
    const f = Math.floor(v), d = v - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  // Modified Bessel function I0 (series), for the Kaiser window
  function besselI0(x) {
    let sum = 1, term = 1;
    const q = (x * x) / 4;
    for (let k = 1; k < 60; k++) { term *= q / (k * k); sum += term; if (term < sum * 1e-17) break; }
    return sum;
  }

  const gcd = (a, b) => (b ? gcd(b, a % b) : a);

  /* libswresample default resampler (filter_size 32, cutoff 0.97, Kaiser beta 9,
   * exact rational phases) — matches `ffmpeg -ar` to ~1e-8. */
  function swrResample(x, rin, rout) {
    const g = gcd(rin, rout), P = rout / g, D = rin / g;
    const factor = Math.min((rout * 0.97) / rin, 1);
    const L = Math.max(Math.ceil(32 / factor), 1), center = (L - 1) >> 1;
    const H = new Float64Array(P * L);
    let norm = 0;
    for (let ph = 0; ph < P; ph++) {
      for (let i = 0; i < L; i++) {
        const xx = Math.PI * ((i - center) - ph / P) * factor;
        let y = xx === 0 ? 1 : Math.sin(xx) / xx;
        const w = (2 * xx) / (factor * L * Math.PI);
        y *= besselI0(9 * Math.sqrt(Math.max(1 - w * w, 0)));
        H[ph * L + i] = y;
        if (ph === 0) norm += y;
      }
    }
    for (let i = 0; i < H.length; i++) H[i] /= norm;
    const N = Math.ceil((x.length * P) / D);
    const out = new Float32Array(N);
    for (let k = 0; k < N; k++) {
      const pos = k * D, idx = Math.floor(pos / P) - center, ph = pos % P, base = ph * L;
      let s = 0;
      const i0 = Math.max(0, -idx), i1 = Math.min(L, x.length - idx);
      for (let i = i0; i < i1; i++) s += H[base + i] * x[idx + i];
      out[k] = s;
    }
    return out;
  }

  /* Read the audio sample rate from the MP4/MOV sample description (mp4a / Opus / ac-3 ...).
   * WebM/Opus always decodes at 48 kHz. Returns 0 if unknown. */
  function sniffAudioRate(u8, name) {
    if (/\.webm$/i.test(name || "")) return 48000;
    const valid = new Set([8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000]);
    const codes = ["mp4a", "Opus", "ac-3", "ec-3", "alac", "fLaC", "sowt", "twos", "lpcm"].map((c) => [...c].map((ch) => ch.charCodeAt(0)));
    for (let p = 4; p < u8.length - 32; p++) {
      for (const c of codes) {
        if (u8[p] !== c[0] || u8[p + 1] !== c[1] || u8[p + 2] !== c[2] || u8[p + 3] !== c[3]) continue;
        const size = (u8[p - 4] << 24) | (u8[p - 3] << 16) | (u8[p - 2] << 8) | u8[p - 1];
        if (size < 36 || size > 4096) continue;
        const rate = (u8[p + 28] << 8) | u8[p + 29];
        if (c[0] === 79 /* Opus */) return 48000;
        if (valid.has(rate)) return rate;
      }
    }
    return 0;
  }

  function once(target, ev, timeout, msg) {
    return new Promise((res, rej) => {
      const t = setTimeout(() => { cleanup(); rej(new Error(msg || "timeout")); }, timeout);
      const ok = () => { cleanup(); res(); };
      const bad = () => { cleanup(); rej(new Error(msg || "media error")); };
      const cleanup = () => { clearTimeout(t); target.removeEventListener(ev, ok); target.removeEventListener("error", bad); };
      target.addEventListener(ev, ok, { once: true });
      target.addEventListener("error", bad, { once: true });
    });
  }
  function seek(v, t) {
    return new Promise((res, rej) => {
      const to = setTimeout(() => { v.removeEventListener("seeked", h); rej(new Error("Seeking the video timed out.")); }, 8000);
      const h = () => { clearTimeout(to); res(); };
      v.addEventListener("seeked", h, { once: true });
      v.currentTime = t;
    });
  }

  window.VerifaiEngine = Engine;
})();
