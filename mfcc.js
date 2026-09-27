/* VerifAI — librosa-compatible MFCC in plain JavaScript
 *
 * Reproduces, step for step, what backend/app/services/preprocess_service.py does:
 *   librosa.feature.mfcc(y, sr=16000, n_mfcc=40)
 *     = melspectrogram(n_fft=2048, hop=512, hann, center=True, pad_mode="constant",
 *                      power=2, n_mels=128, fmin=0, fmax=sr/2, Slaney mel scale + norm)
 *     -> power_to_db(ref=1, amin=1e-10, top_db=80)
 *     -> DCT-II (orthonormal), first 40 coefficients
 *   then pad/trim to 200 frames and standardise with torch's (unbiased) std.
 *
 * Verified against librosa 0.11 by scripts/check_mfcc_parity (see repo).
 * © 2026 Giulio Labs
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.VerifaiMFCC = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const hzToMel = (f) => {
    const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27;
    return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fSp;
  };
  const melToHz = (m) => {
    const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27;
    return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m;
  };

  // librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, htk=False, norm="slaney") -> float32
  function melFilterbank(sr, nFft, nMels) {
    const nBins = 1 + (nFft >> 1);
    const fftFreqs = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) fftFreqs[k] = (k * (sr / 2)) / (nBins - 1);
    const mMin = hzToMel(0), mMax = hzToMel(sr / 2);
    const melF = new Float64Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) melF[i] = melToHz(mMin + ((mMax - mMin) * i) / (nMels + 1));
    const fb = [];
    for (let i = 0; i < nMels; i++) {
      const row = new Float32Array(nBins);
      const d0 = melF[i + 1] - melF[i], d1 = melF[i + 2] - melF[i + 1];
      const enorm = 2 / (melF[i + 2] - melF[i]);
      let lo = -1, hi = -1;
      for (let k = 0; k < nBins; k++) {
        const lower = -(melF[i] - fftFreqs[k]) / d0;
        const upper = (melF[i + 2] - fftFreqs[k]) / d1;
        const w = Math.max(0, Math.min(lower, upper));
        if (w > 0) { row[k] = Math.fround(w * enorm); if (lo < 0) lo = k; hi = k; }
      }
      fb.push({ row, lo: Math.max(lo, 0), hi });
    }
    return fb;
  }

  // In-place iterative radix-2 complex FFT
  function makeFFT(n) {
    const levels = Math.log2(n) | 0;
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < levels; b++) r |= ((i >>> b) & 1) << (levels - 1 - b);
      rev[i] = r;
    }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos((2 * Math.PI * i) / n); sin[i] = Math.sin((2 * Math.PI * i) / n); }
    return function fft(re, im) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
      }
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1, step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = i, k = 0; j < i + half; j++, k += step) {
            const tRe = re[j + half] * cos[k] + im[j + half] * sin[k];
            const tIm = -re[j + half] * sin[k] + im[j + half] * cos[k];
            re[j + half] = re[j] - tRe; im[j + half] = im[j] - tIm;
            re[j] += tRe; im[j] += tIm;
          }
        }
      }
    };
  }

  const cache = {};

  /**
   * @param {Float32Array} y mono audio at `sr`
   * @returns {{mfcc: Float32Array, frames: number}} mfcc is row-major [nMfcc, frames]
   */
  function mfcc(y, { sr = 16000, nMfcc = 40, nFft = 2048, hop = 512, nMels = 128, topDb = 80 } = {}) {
    const key = `${sr}|${nFft}|${nMels}`;
    const fb = cache[key] || (cache[key] = melFilterbank(sr, nFft, nMels));
    const nBins = 1 + (nFft >> 1);
    const pad = nFft >> 1;
    const nFrames = 1 + Math.floor(y.length / hop);
    const win = new Float64Array(nFft);
    for (let i = 0; i < nFft; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft); // periodic Hann
    const fft = makeFFT(nFft);
    const re = new Float64Array(nFft), im = new Float64Array(nFft), pw = new Float64Array(nBins);

    // mel power spectrogram -> dB
    const melDb = new Float64Array(nMels * nFrames);
    let maxDb = -Infinity;
    for (let t = 0; t < nFrames; t++) {
      const start = t * hop - pad;
      for (let i = 0; i < nFft; i++) {
        const idx = start + i;
        re[i] = idx >= 0 && idx < y.length ? y[idx] * win[i] : 0;
        im[i] = 0;
      }
      fft(re, im);
      for (let k = 0; k < nBins; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
      for (let m = 0; m < nMels; m++) {
        const { row, lo, hi } = fb[m];
        let s = 0;
        for (let k = lo; k <= hi; k++) s += row[k] * pw[k];
        const db = 10 * Math.log10(Math.max(1e-10, s));
        melDb[m * nFrames + t] = db;
        if (db > maxDb) maxDb = db;
      }
    }
    const floor = maxDb - topDb;
    for (let i = 0; i < melDb.length; i++) if (melDb[i] < floor) melDb[i] = floor;

    // DCT-II, orthonormal, along the mel axis
    const out = new Float32Array(nMfcc * nFrames);
    const s0 = Math.sqrt(1 / nMels), s1 = Math.sqrt(2 / nMels);
    const basis = new Float64Array(nMfcc * nMels);
    for (let c = 0; c < nMfcc; c++)
      for (let m = 0; m < nMels; m++)
        basis[c * nMels + m] = Math.cos((Math.PI * c * (2 * m + 1)) / (2 * nMels)) * (c === 0 ? s0 : s1);
    for (let t = 0; t < nFrames; t++)
      for (let c = 0; c < nMfcc; c++) {
        let s = 0;
        for (let m = 0; m < nMels; m++) s += basis[c * nMels + m] * melDb[m * nFrames + t];
        out[c * nFrames + t] = s;
      }
    return { mfcc: out, frames: nFrames };
  }

  /** mfcc -> [1,1,nMfcc,maxLen] model input (pad/trim + per-sample standardisation) */
  function modelInput(y, { nMfcc = 40, maxLen = 200, sr = 16000 } = {}) {
    if (!y || y.length === 0) return null;
    const { mfcc: m, frames } = mfcc(y, { sr, nMfcc });
    const x = new Float32Array(nMfcc * maxLen);
    for (let c = 0; c < nMfcc; c++)
      for (let t = 0; t < Math.min(frames, maxLen); t++) x[c * maxLen + t] = m[c * frames + t];
    let mean = 0;
    for (let i = 0; i < x.length; i++) mean += x[i];
    mean /= x.length;
    let v = 0;
    for (let i = 0; i < x.length; i++) v += (x[i] - mean) ** 2;
    const std = Math.max(Math.sqrt(v / (x.length - 1)), 1e-6); // torch.std is unbiased
    let allZero = true;
    for (let i = 0; i < x.length; i++) { x[i] = (x[i] - mean) / std; if (x[i] !== 0) allZero = false; }
    return allZero ? null : x; // backend treats an all-zero MFCC as "no usable audio"
  }

  return { mfcc, modelInput, melFilterbank };
});
