export const N_MFCC = 13;
export const N_MELS = 26;

export function autocorrelationF0(
  buffer: Float32Array,
  sampleRate: number,
): { f0: number; confidence: number } {
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
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
  return Math.sqrt(sum / buffer.length);
}

export function computeSpectralCentroid(fft: Float32Array, sampleRate: number): number {
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

export function buildMelFilterbank(fftSize: number, sampleRate: number): Float32Array[] {
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
  return filterbank;
}

export function computeMFCC(magSpectrum: Float32Array, filterbank: Float32Array[]): number[] {
  const melEnergies = new Float32Array(N_MELS);
  for (let i = 0; i < N_MELS; i++) {
    let energy = 0;
    const filter = filterbank[i];
    for (let j = 0; j < Math.min(magSpectrum.length, filter.length); j++) {
      energy += magSpectrum[j] * filter[j];
    }
    melEnergies[i] = Math.log(Math.max(energy, 1e-10));
  }

  const mfcc = new Array<number>(N_MFCC);
  for (let k = 0; k < N_MFCC; k++) {
    let sum = 0;
    for (let n = 0; n < N_MELS; n++) {
      sum += melEnergies[n] * Math.cos((Math.PI * k * (n + 0.5)) / N_MELS);
    }
    mfcc[k] = sum;
  }
  return mfcc;
}
