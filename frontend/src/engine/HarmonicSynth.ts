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
  private dryGain: GainNode;
  private wetGain: GainNode;
  private convolver: ConvolverNode;
  private outputGain: GainNode;
  
  private ctx: AudioContext;
  private prevF0 = 0;
  private melFb: Float32Array[] | null = null;

  constructor(ctx: AudioContext, destination: AudioNode) {
    this.ctx = ctx;
    
    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = 0;
    
    // Setup Dry/Wet Reverb Mix
    this.dryGain = ctx.createGain();
    this.wetGain = ctx.createGain();
    this.convolver = ctx.createConvolver();
    this.outputGain = ctx.createGain();
    this.outputGain.gain.value = 1.0;

    this.dryGain.gain.value = 1.0;
    this.wetGain.gain.value = 0.0; // fully dry until DDSP drives it

    this.masterGain.connect(this.dryGain);
    this.masterGain.connect(this.convolver);
    this.convolver.connect(this.wetGain);
    
    this.dryGain.connect(this.outputGain);
    this.wetGain.connect(this.outputGain);
    this.outputGain.connect(destination);

    this._initPartials();
    this._initNoise();
  }

  async loadReverb(url: string = '/models/reverb_ir.wav') {
    try {
      const response = await fetch(url);
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await this.ctx.decodeAudioData(arrayBuffer);
      this.convolver.buffer = audioBuffer;
      console.log(`[DDSP Reverb] Loaded IR from ${url}`);
    } catch (e) {
      console.warn(`[DDSP Reverb] Reverb IR not found at ${url}, running completely dry.`);
    }
  }

  setMuted(muted: boolean) {
    const now = this.ctx.currentTime;
    this.outputGain.gain.setTargetAtTime(muted ? 0.0 : 1.0, now, 0.05); // 50ms smooth fade
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
    const smooth = 0.02;

    if (frame.f0Hz > 20 && frame.f0Hz < 2000 && frame.loudnessNorm > 0.005) {
      const N_ACTIVE = 24; // 24 mel-weighted partials vs old 8-fixed — dramatically richer timbre
      const melBands = frame.melBands as Float32Array | undefined;
      const sampleRate = this.ctx.sampleRate;
      const nyquist = sampleRate / 2;

      for (let h = 0; h < N_ACTIVE; h++) {
        const freq = frame.f0Hz * (h + 1);
        if (freq >= nyquist * 0.98) {
          // Anti-alias: zero partials at or above Nyquist
          this.gains[h].gain.setTargetAtTime(0, now, smooth);
          continue;
        }

        this.partials[h].frequency.setTargetAtTime(freq, now, smooth);

        // Mel-band-informed amplitude: map each partial's frequency to the nearest mel band
        let amp: number;
        if (melBands && melBands.length >= 64) {
          // Map freq → mel band index
          const melFreq = 2595 * Math.log10(1 + freq / 700);
          const melMax = 2595 * Math.log10(1 + nyquist / 700);
          const bandIdx = Math.min(63, Math.max(0, Math.round((melFreq / melMax) * 63)));
          // Log-mel values are in dB range [-80, 0]; normalize to [0, 1]
          const normalizedEnergy = Math.max(0, (melBands[bandIdx] + 80) / 80);
          // Natural rolloff weighted by mel energy
          amp = normalizedEnergy / Math.pow(h + 1, 0.6);
        } else {
          // Pure spectral rolloff fallback (no mel data available)
          amp = 1 / Math.pow(h + 1, 0.8);
        }

        this.gains[h].gain.setTargetAtTime(amp * 0.15, now, smooth);
      }

      // Silence partials 24–59 (upper register, not used in fallback mode)
      for (let h = N_ACTIVE; h < N_PARTIALS; h++) {
        this.gains[h].gain.setTargetAtTime(0, now, smooth);
      }

      // Perceptual loudness: cube-root scaling approximates perceived loudness better than linear
      const perceivedLoudness = Math.pow(Math.min(1, Math.max(0, frame.loudnessNorm)), 1 / 3);
      this.masterGain.gain.setTargetAtTime(perceivedLoudness * 0.7, now, smooth);
    } else {
      this.masterGain.gain.setTargetAtTime(0, now, 0.05);
    }

    this.prevF0 = frame.f0Hz;
  }


  setTarget(f0Hz: number, rmsEnergy: number, harmonics: Float32Array, noiseMagnitudes?: Float32Array, reverbMix?: Float32Array, f0Residual?: Float32Array, voicedMask?: Float32Array) {
    const now = this.ctx.currentTime;
    
    // Apply pitch error correction (native f0 bend residual predicted by DDSP)
    if (f0Residual && f0Residual.length > 0) {
      f0Hz = f0Hz * (1.0 + f0Residual[0]);
    }

    const smooth = Math.abs(f0Hz - this.prevF0) > 50 ? 0.005 : 0.05;

    if (f0Hz > 0 && rmsEnergy > 0.01) {
      const vGate = (voicedMask && voicedMask.length > 0) ? voicedMask[0] : 1.0;

      for (let h = 0; h < N_PARTIALS; h++) {
        const hFreq = f0Hz * (h + 1);
        // Anti-aliasing check (Nyquist limit)
        if (hFreq < this.ctx.sampleRate / 2) {
          this.partials[h].frequency.setTargetAtTime(hFreq, now, smooth);
          const amp = h < harmonics.length ? harmonics[h] : 0;
          
          // Apply explicit Voiced/Unvoiced gate to mute partials during consonants
          this.gains[h].gain.setTargetAtTime(amp * 0.15 * vGate, now, smooth);
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

      // Reverb Mixing
      if (reverbMix && reverbMix.length > 0 && this.convolver.buffer) {
        // Average the mix over the frame
        const mix = reverbMix[0]; 
        this.dryGain.gain.setTargetAtTime(Math.max(0, 1.0 - mix), now, smooth);
        this.wetGain.gain.setTargetAtTime(Math.max(0, mix), now, smooth);
      } else {
        this.dryGain.gain.setTargetAtTime(1.0, now, smooth);
        this.wetGain.gain.setTargetAtTime(0.0, now, smooth);
      }
    } else {
      this.masterGain.gain.setTargetAtTime(0, now, 0.05);
      this.dryGain.gain.setTargetAtTime(1.0, now, 0.05);
      this.wetGain.gain.setTargetAtTime(0.0, now, 0.05);
    }

    this.prevF0 = f0Hz;
  }

  destroy() {
    this.partials.forEach(o => { try { o.stop(); } catch (_) {} });
    this.noiseSource?.stop();
    this.masterGain.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
    this.convolver.disconnect();
    this.outputGain.disconnect();
  }
}
