export const N_MFCC = 13;
export const N_MELS = 64;

// ─── WASM Integration ───────────────────────────────────────────────────────

interface WasmCore {
  _malloc(size: number): number;
  _free(ptr: number): void;
  HEAPF32: Float32Array;
  
  _computeRMS(ptr: number, length: number): number;
  _autocorrelationF0(ptr: number, length: number, sampleRate: number, outF0Ptr: number, outConfPtr: number): void;
  _computeSpectralCentroid(fftPtr: number, length: number, sampleRate: number): number;
  _initFilterbank(fftSize: number, sampleRate: number): void;
  _computeMFCC(magSpectrumPtr: number, spectrumLength: number, outMfccPtr: number): void;
}

let wasmCore: WasmCore | null = null;
let isWasmLoaded = false;

/**
 * Initialize the C++ WebAssembly intent core for sub-millisecond extraction.
 * Expects the Emscripten output module (intent_core.js) to be available globally
 * or injected via Vite.
 */
export async function initWasmDSP(createIntentModule?: any): Promise<void> {
  if (isWasmLoaded) return;
  if (!createIntentModule) {
    console.warn("WASM module factory not provided. Falling back to JS DSP.");
    return;
  }
  
  try {
    const module = await createIntentModule();
    wasmCore = module as WasmCore;
    isWasmLoaded = true;
    console.log("Sub-10ms WASM Intent Core initialized successfully.");
  } catch (err) {
    console.error("Failed to load WASM intent core, falling back to JS", err);
  }
}

// ─── Extraction Wrappers (WASM + JS Fallback) ───────────────────────────────

export function autocorrelationF0(
  buffer: Float32Array,
  sampleRate: number,
): { f0: number; confidence: number } {
  if (isWasmLoaded && wasmCore) {
    const n = buffer.length;
    const ptr = wasmCore._malloc(n * 4);
    const outF0Ptr = wasmCore._malloc(4);
    const outConfPtr = wasmCore._malloc(4);
    
    wasmCore.HEAPF32.set(buffer, ptr / 4);
    wasmCore._autocorrelationF0(ptr, n, sampleRate, outF0Ptr, outConfPtr);
    
    const f0 = wasmCore.HEAPF32[outF0Ptr / 4];
    const confidence = wasmCore.HEAPF32[outConfPtr / 4];
    
    wasmCore._free(ptr);
    wasmCore._free(outF0Ptr);
    wasmCore._free(outConfPtr);
    return { f0, confidence };
  }

  // JS Fallback
  const n = buffer.length;
  const minPeriod = Math.floor(sampleRate / 1046); // C6 max
  const maxPeriod = Math.min(Math.floor(sampleRate / 40), n - 1); // E1 min

  if (maxPeriod <= minPeriod) return { f0: 0, confidence: 0 };

  let bestCorr = 0;
  let bestPeriod = 0;
  let energy = 0;

  for (let i = 0; i < n; i++) energy += buffer[i] * buffer[i];
  const rms = Math.sqrt(energy / n);
  if (rms < 0.005) return { f0: 0, confidence: 0 }; // Silence gate

  for (let period = minPeriod; period <= maxPeriod; period++) {
    let corr = 0;
    let e1 = 0;
    let e2 = 0;
    const len = n - period;
    for (let i = 0; i < len; i++) {
      corr += buffer[i] * buffer[i + period];
      e1 += buffer[i] * buffer[i];
      e2 += buffer[i + period] * buffer[i + period];
    }
    const norm = Math.sqrt(e1 * e2);
    const normCorr = norm > 0 ? corr / norm : 0;

    if (normCorr > bestCorr) {
      bestCorr = normCorr;
      bestPeriod = period;
    }
  }

  if (bestCorr < 0.5 || bestPeriod === 0) return { f0: 0, confidence: 0 };

  return {
    f0: sampleRate / bestPeriod,
    confidence: bestCorr,
  };
}

export function computeRMS(buffer: Float32Array): number {
  if (isWasmLoaded && wasmCore) {
    const ptr = wasmCore._malloc(buffer.length * 4);
    wasmCore.HEAPF32.set(buffer, ptr / 4);
    const rms = wasmCore._computeRMS(ptr, buffer.length);
    wasmCore._free(ptr);
    return rms;
  }

  // JS Fallback
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
  return Math.sqrt(sum / buffer.length);
}

export function computeSpectralCentroid(fft: Float32Array, sampleRate: number): number {
  if (isWasmLoaded && wasmCore) {
    const ptr = wasmCore._malloc(fft.length * 4);
    wasmCore.HEAPF32.set(fft, ptr / 4);
    const centroid = wasmCore._computeSpectralCentroid(ptr, fft.length, sampleRate);
    wasmCore._free(ptr);
    return centroid;
  }

  // JS Fallback
  let weightedSum = 0;
  let totalMag = 0;
  const binWidth = sampleRate / (fft.length * 2);
  for (let i = 0; i < fft.length; i++) {
    const mag = Math.abs(fft[i]);
    weightedSum += mag * (i * binWidth);
    totalMag += mag;
  }
  return totalMag > 0 ? weightedSum / totalMag : 0;
}

// Global filterbank state for JS Fallback
let jsFilterbank: Float32Array[] | null = null;

export function buildMelFilterbank(fftSize: number, sampleRate: number): void {
  if (isWasmLoaded && wasmCore) {
    wasmCore._initFilterbank(fftSize, sampleRate);
    return;
  }

  // JS Fallback
  const numBins = fftSize / 2;
  const maxFreq = sampleRate / 2;
  const maxMel = 2595 * Math.log10(1 + maxFreq / 700);

  const melPoints = new Float32Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i++) {
    const mel = (i * maxMel) / (N_MELS + 1);
    melPoints[i] = 700 * (Math.pow(10, mel / 2595) - 1);
  }

  const binPoints = new Int32Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i++) {
    binPoints[i] = Math.floor((numBins * melPoints[i]) / maxFreq);
  }

  const filterbank: Float32Array[] = [];
  for (let i = 0; i < N_MELS; i++) {
    const filter = new Float32Array(numBins);
    for (let j = binPoints[i]; j < binPoints[i + 1]; j++) {
      filter[j] = (j - binPoints[i]) / (binPoints[i + 1] - binPoints[i] + 1e-10);
    }
    for (let j = binPoints[i + 1]; j < binPoints[i + 2]; j++) {
      filter[j] = (binPoints[i + 2] - j) / (binPoints[i + 2] - binPoints[i + 1] + 1e-10);
    }
    filterbank.push(filter);
  }
  jsFilterbank = filterbank;
}

export function computeMFCC(magSpectrum: Float32Array, ignoreFilterbankParam?: any): number[] {
  // Deprecated - kept for structural safety if modules are still bound
  return new Array(N_MFCC).fill(0);
}

export function computeMelBands(magSpectrum: Float32Array): Float32Array {
  // Always use JS Fallback for 64-dim Log-Mels until WASM module is re-compiled
  if (!jsFilterbank) return new Float32Array(N_MELS);

  const melEnergies = new Float32Array(N_MELS);
  for (let i = 0; i < N_MELS; i++) {
    let energy = 0;
    const filter = jsFilterbank[i];
    for (let j = 0; j < Math.min(magSpectrum.length, filter.length); j++) {
      // Mag spectrum is magnitude, so we square it to get power for power_to_db equivalent
      energy += (magSpectrum[j] * magSpectrum[j]) * filter[j];
    }
    // 10 * log10 matches Librosa's power_to_db
    melEnergies[i] = 10 * Math.log10(Math.max(energy, 1e-10));
  }

  return melEnergies;
}
