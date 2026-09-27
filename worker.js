/* VerifAI — inference worker (ONNX Runtime Web, WebAssembly)
 * Runs Xception (visual) and ResNet-18 (MFCC audio) entirely on the visitor's device.
 * © 2026 Giulio Labs
 */
/* global ort */
"use strict";

const ORT_VERSION = "1.30.0";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
importScripts(`${ORT_CDN}ort.wasm.min.js`);

ort.env.wasm.wasmPaths = ORT_CDN;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
ort.env.logLevel = "error";

const CACHE_NAME = "verifai-models-v1";
let sessions = null;
let loading = null;

async function fetchWithProgress(url, onBytes) {
  // Cache Storage first — makes every visit after the first instant.
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { cache = null; }
  if (cache) {
    const hit = await cache.match(url).catch(() => null);
    if (hit) { const buf = await hit.arrayBuffer(); onBytes(buf.byteLength, buf.byteLength, true); return buf; }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${url.split("/").pop()} (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  let buf;
  if (res.body && res.body.getReader) {
    const reader = res.body.getReader();
    const chunks = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.byteLength; onBytes(got, total, false);
    }
    const out = new Uint8Array(got); let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.byteLength; }
    buf = out.buffer;
  } else {
    buf = await res.arrayBuffer();
  }
  if (cache) cache.put(url, new Response(buf.slice(0), { headers: { "content-type": "application/octet-stream" } })).catch(() => {});
  return buf;
}

async function load(base, files) {
  if (sessions) return sessions;
  if (loading) return loading;
  loading = (async () => {
    const got = {}, totals = {};
    const report = () => {
      const g = Object.values(got).reduce((a, b) => a + b, 0);
      const t = files.reduce((a, f) => a + (totals[f.name] || f.approxBytes), 0);
      postMessage({ type: "progress", loaded: g, total: t });
    };
    const bufs = await Promise.all(files.map((f) =>
      fetchWithProgress(base + f.name, (g, t) => { got[f.name] = g; if (t) totals[f.name] = t; report(); })));
    postMessage({ type: "status", text: "Initialising models…" });
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    const out = {};
    for (let i = 0; i < files.length; i++) out[files[i].key] = await ort.InferenceSession.create(bufs[i], opts);
    sessions = out;
    return out;
  })();
  try { return await loading; } catch (e) { loading = null; throw e; }
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await load(msg.base, msg.files);
      postMessage({ type: "loaded", threads: ort.env.wasm.numThreads });
    } else if (msg.type === "infer") {
      const s = await load(msg.base, msg.files);
      const t0 = performance.now();
      const n = msg.frames.length / (3 * 224 * 224);
      const vis = await s.visual.run({ frames: new ort.Tensor("float32", msg.frames, [n, 3, 224, 224]) });
      const visualLogits = Array.from(vis.logits.data);
      postMessage({ type: "stage", stage: "audio" });
      let audioLogit = null;
      if (msg.mfcc) {
        const aud = await s.audio.run({ mfcc: new ort.Tensor("float32", msg.mfcc, [1, 1, 40, 200]) });
        audioLogit = aud.logit.data[0];
      }
      postMessage({ type: "result", id: msg.id, visualLogits, audioLogit, ms: performance.now() - t0 });
    }
  } catch (err) {
    postMessage({ type: "error", id: msg.id, message: String((err && err.message) || err) });
  }
};
