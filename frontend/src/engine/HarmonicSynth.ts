import { IntentFrame } from './types';
import { N_MELS } from './dsp';

const N_PARTIALS = 60;

export class HarmonicSynth {
  private partials: OscillatorNode[] = [];
  private gains: GainNode[] = [];
  private noiseSource: AudioBufferSourceNode | null = null;
  private noiseBands: BiquadFilterNode[] = [];
  private noiseGains: GainNode[] = [];
  private readonly N_NOISE_BANDS = 16;
  private masterGain: GainNode;
  private ctx: AudioContext;
  private prevF0 = 0;
  private melFb: Float32Array[] | null = null;

  constructor(ctx: AudioContext, destination: AudioNode) {
    this.ctx = ctx;
    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = 0;
    this.masterGain.connect(destination);
    this._initPartials();
    this._initNoise();
  }

  setMelFilterbank(fb: Float32Array[]) { this.melFb = fb; }

  private _initPartials() {
    for (let i = 0; i < N_PARTIALS; i++) {
      const osc = this.ctx.createOscillator();
      osc.type = 'sine';
      const g = this.ctx.createGain();
      g.gain.value = 0;
      osc.connect(g);
      g.connect(this.masterGain);
      osc.start();
      this.partials.push(osc);
      this.gains.push(g);
    }
  }

  private _initNoise() {
    // 2-second white noise buffer, looped — avoids ScriptProcessor overhead
    const bufSize = this.ctx.sampleRate * 2;
    const noiseBuffer = this.ctx.createBuffer(1, bufSize, this.ctx.sampleRate);
    const data = noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufSize; i++) data[i] = Math.random() * 2 - 1;

    this.noiseSource = this.ctx.createBufferSource();
    this.noiseSource.buffer = noiseBuffer;
    this.noiseSource.loop = true;

    const nyquist = this.ctx.sampleRate / 2;
    const bandwidth = nyquist / this.N_NOISE_BANDS;

    for (let i = 0; i < this.N_NOISE_BANDS; i++) {
        const filter = this.ctx.createBiquadFilter();
        // Lowest band is lowpass, rest are bandpass
        filter.type = i === 0 ? 'lowpass' : 'bandpass';
        
        const centerFreq = nyquist * ((i + 0.5) / this.N_NOISE_BANDS);
        filter.frequency.value = centerFreq;
        filter.Q.value = i === 0 ? 0.7 : (centerFreq / bandwidth);

        const gain = this.ctx.createGain();
        gain.gain.value = 0;

        this.noiseSource.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);

        this.noiseBands.push(filter);
        this.noiseGains.push(gain);
    }

    this.noiseSource.start();
  }

  /** Convert MFCC array → per-partial amplitude via inverse DCT + mel mapping */
  private _mfccToAmps(mfcc: number[], f0Hz: number): number[] {
    const amps = new Array(N_PARTIALS).fill(0);
    // Inverse DCT → mel log-energy envelope (N_MELS bands)
    const melEnv = Array.from({ length: N_MELS }, (_, j) =>
      mfcc.reduce((acc, c, i) => acc + c * Math.cos(Math.PI * i * (j + 0.5) / N_MELS), 0)
    );
    const maxEnv = Math.max(...melEnv.map(Math.abs)) + 1e-10;
    const normEnv = melEnv.map(v => Math.exp(v / maxEnv)); // positive envelope
    const envMax = Math.max(...normEnv) + 1e-10;
    const scaledEnv = normEnv.map(v => v / envMax);

    const sr = this.ctx.sampleRate;
    const melMax = 2595 * Math.log10(1 + (sr / 2) / 700);
    for (let h = 0; h < N_PARTIALS; h++) {
      const freq = f0Hz * (h + 1);
      if (freq >= sr / 2) break;
      const mel = 2595 * Math.log10(1 + freq / 700);
      const band = (mel / melMax) * (N_MELS - 1);
      const lo = Math.floor(band), hi = Math.min(lo + 1, N_MELS - 1);
      const frac = band - lo;
      const envAmp = scaledEnv[lo] * (1 - frac) + scaledEnv[hi] * frac;
      // Natural rolloff: higher harmonics are quieter
      amps[h] = envAmp / Math.pow(h + 1, 0.5);
    }
    const maxAmp = Math.max(...amps) + 1e-10;
    return amps.map(a => a / maxAmp);
  }

  update(frame: IntentFrame) {
    const now = this.ctx.currentTime;
    const smooth = 0.015; // 15ms smoothing

    if (frame.f0Hz > 0 && frame.confidence > 0.45 && frame.loudnessNorm > 0.01) {
      const amps = frame.mfcc.some(v => v !== 0)
        ? this._mfccToAmps(frame.mfcc, frame.f0Hz)
        : Array.from({ length: N_PARTIALS }, (_, h) => 1 / Math.pow(h + 1, 0.6));

      for (let h = 0; h < N_PARTIALS; h++) {
        this.partials[h].frequency.setTargetAtTime(frame.f0Hz * (h + 1), now, smooth);
        this.gains[h].gain.setTargetAtTime(amps[h] * 0.15, now, smooth);
      }

      // Set master loudness
      this.masterGain.gain.setTargetAtTime(frame.loudnessNorm * 0.7, now, smooth);

      // Shape noise: spectral centroid drives filter cutoff
      if (this.noiseBands.length > 0) {
        const cutoff = Math.min(Math.max(frame.spectralCentroid * 1.5, 200), 8000);
        const noiseLevel = (1 - frame.confidence) * 0.08 * frame.loudnessNorm;
        for (let i = 0; i < this.N_NOISE_BANDS; i++) {
            const center = this.noiseBands[i].frequency.value;
            const rollOff = Math.max(0, 1 - (center / cutoff));
            this.noiseGains[i].gain.setTargetAtTime(noiseLevel * rollOff, now, smooth);
        }
      }

      // Onset transient: brief gain burst
      if (frame.isOnset && frame.onsetStrength > 0.05) {
        this.masterGain.gain.setValueAtTime(frame.loudnessNorm * 1.5, now);
        this.masterGain.gain.setTargetAtTime(frame.loudnessNorm * 0.7, now + 0.005, 0.02);
      }

    } else {
      // Silence: fade out
      this.masterGain.gain.setTargetAtTime(0, now, 0.05);
    }

    this.prevF0 = frame.f0Hz;
  }

  setTarget(f0Hz: number, rmsEnergy: number, harmonics: Float32Array | number[], noiseMagnitudes?: Float32Array | number[]) {
    const now = this.ctx.currentTime;
    const smooth = 0.015; // 15ms smoothing

    if (f0Hz > 0 && rmsEnergy > 0.01) {
      for (let h = 0; h < N_PARTIALS; h++) {
        const hFreq = f0Hz * (h + 1);
        // Anti-aliasing check (Nyquist limit)
        if (hFreq < this.ctx.sampleRate / 2) {
          this.partials[h].frequency.setTargetAtTime(hFreq, now, smooth);
          const amp = h < harmonics.length ? harmonics[h] : 0;
          this.gains[h].gain.setTargetAtTime(amp * 0.15, now, smooth);
        } else {
          this.gains[h].gain.setTargetAtTime(0, now, smooth);
        }
      }

      this.masterGain.gain.setTargetAtTime(rmsEnergy * 0.7, now, smooth);

      // Highly-detailed Neural Noise mapping
      if (noiseMagnitudes && noiseMagnitudes.length > 0) {
        const ratio = noiseMagnitudes.length / this.N_NOISE_BANDS;
        for (let b = 0; b < this.N_NOISE_BANDS; b++) {
            const startBin = Math.floor(b * ratio);
            const endBin = Math.floor((b + 1) * ratio);
            let sum = 0;
            for (let j = startBin; j < endBin; j++) sum += noiseMagnitudes[j];
            const avgMag = sum / (endBin - startBin + 1e-6);
            
            // Map the band magnitude to subjective gain
            const gain = rmsEnergy * avgMag;
            this.noiseGains[b].gain.setTargetAtTime(gain, now, smooth);
        }
      } else if (this.noiseBands.length > 0) {
        const baseLevel = Math.max(0, rmsEnergy * 0.02);
        for (let i = 0; i < this.N_NOISE_BANDS; i++) {
             const roll = 1.0 / (1.0 + i); 
             this.noiseGains[i].gain.setTargetAtTime(baseLevel * roll, now, smooth);
        }
      }
    } else {
      this.masterGain.gain.setTargetAtTime(0, now, 0.05);
    }

    this.prevF0 = f0Hz;
  }

  destroy() {
    this.partials.forEach(o => { try { o.stop(); } catch (_) {} });
    this.noiseSource?.stop();
    this.masterGain.disconnect();
  }
}
