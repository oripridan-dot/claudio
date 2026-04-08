export type OscillatorType = 'sine' | 'square' | 'sawtooth' | 'triangle';

export interface EffectParams {
  reverb: number;
  reverbDecay: number;
  delay: number;
  delayTime: number;
  delayFeedback: number;
  distortion: number;
  compThreshold: number;
  compRatio: number;
  masterGain: number;
}

/**
 * Claudio AudioEngine — real-time Web Audio API DSP chain.
 *
 * Signal path:
 *   Source → InputGain → WaveShaper (distortion) → DynamicsCompressor
 *     → Delay (dry/wet parallel) → Reverb (dry/wet parallel)
 *     → MasterGain → [SpectrumAnalyser, WaveformAnalyser] → Destination
 */
export class AudioEngine {
  readonly ctx: AudioContext;

  // Nodes
  private inputGain: GainNode;
  private waveshaper: WaveShaperNode;
  private compressor: DynamicsCompressorNode;
  private delayNode: DelayNode;
  private delayFeedbackGain: GainNode;
  private delayDryGain: GainNode;
  private delayWetGain: GainNode;
  private delayMix: GainNode;
  private reverbConvolver: ConvolverNode;
  private reverbDryGain: GainNode;
  private reverbWetGain: GainNode;
  private masterGainNode: GainNode;

  // Public analysers
  readonly spectrumAnalyser: AnalyserNode;
  readonly waveformAnalyser: AnalyserNode;

  // Sources
  private oscillator: OscillatorNode | null = null;
  private micSource: MediaStreamAudioSourceNode | null = null;
  private micStream: MediaStream | null = null;

  private _oscActive = false;
  private _micActive = false;

  params: EffectParams = {
    reverb: 0.15,
    reverbDecay: 2.0,
    delay: 0.0,
    delayTime: 0.33,
    delayFeedback: 0.35,
    distortion: 0.0,
    compThreshold: -24,
    compRatio: 4,
    masterGain: 0.75,
  };

  constructor() {
    const AC = window.AudioContext || (window as any).webkitAudioContext;
    this.ctx = new AC();

    // ── Build nodes ──────────────────────────────────────────────────────
    this.inputGain = this.ctx.createGain();
    this.inputGain.gain.value = 1.0;

    this.waveshaper = this.ctx.createWaveShaper();
    this.waveshaper.oversample = '4x';
    this._buildDistortionCurve(0);

    this.compressor = this.ctx.createDynamicsCompressor();
    this.compressor.threshold.value = this.params.compThreshold;
    this.compressor.ratio.value = this.params.compRatio;
    this.compressor.attack.value = 0.003;
    this.compressor.release.value = 0.25;
    this.compressor.knee.value = 3;

    this.delayNode = this.ctx.createDelay(2.0);
    this.delayNode.delayTime.value = this.params.delayTime;
    this.delayFeedbackGain = this.ctx.createGain();
    this.delayFeedbackGain.gain.value = this.params.delayFeedback;
    this.delayDryGain = this.ctx.createGain();
    this.delayDryGain.gain.value = 1.0;
    this.delayWetGain = this.ctx.createGain();
    this.delayWetGain.gain.value = 0.0;
    this.delayMix = this.ctx.createGain();
    this.delayMix.gain.value = 1.0;

    this.reverbConvolver = this.ctx.createConvolver();
    this._buildImpulseResponse(this.params.reverbDecay);
    this.reverbDryGain = this.ctx.createGain();
    this.reverbDryGain.gain.value = 1 - this.params.reverb;
    this.reverbWetGain = this.ctx.createGain();
    this.reverbWetGain.gain.value = this.params.reverb;

    this.masterGainNode = this.ctx.createGain();
    this.masterGainNode.gain.value = this.params.masterGain;

    this.spectrumAnalyser = this.ctx.createAnalyser();
    this.spectrumAnalyser.fftSize = 2048;
    this.spectrumAnalyser.smoothingTimeConstant = 0.82;

    this.waveformAnalyser = this.ctx.createAnalyser();
    this.waveformAnalyser.fftSize = 1024;
    this.waveformAnalyser.smoothingTimeConstant = 0.0;

    this._wire();
  }

  private _wire(): void {
    // inputGain → distortion → compressor
    this.inputGain.connect(this.waveshaper);
    this.waveshaper.connect(this.compressor);

    // Delay: compressor → dry + (wet via feedback loop) → mix
    this.compressor.connect(this.delayDryGain);
    this.compressor.connect(this.delayNode);
    this.delayNode.connect(this.delayFeedbackGain);
    this.delayFeedbackGain.connect(this.delayNode);  // feedback loop
    this.delayNode.connect(this.delayWetGain);
    this.delayDryGain.connect(this.delayMix);
    this.delayWetGain.connect(this.delayMix);

    // Reverb: delayMix → dry + convolverWet → masterGain
    this.delayMix.connect(this.reverbDryGain);
    this.delayMix.connect(this.reverbConvolver);
    this.reverbConvolver.connect(this.reverbWetGain);
    this.reverbDryGain.connect(this.masterGainNode);
    this.reverbWetGain.connect(this.masterGainNode);

    // masterGain → analysers + output
    this.masterGainNode.connect(this.spectrumAnalyser);
    this.masterGainNode.connect(this.waveformAnalyser);
    this.masterGainNode.connect(this.ctx.destination);
  }

  private _buildDistortionCurve(amount: number): void {
    const n = 256;
    const curve = new Float32Array(n);
    const k = amount * 200;
    for (let i = 0; i < n; i++) {
      const x = (i * 2) / n - 1;
      curve[i] = k > 0
        ? ((Math.PI + k) * x) / (Math.PI + k * Math.abs(x))
        : x;
    }
    this.waveshaper.curve = curve;
  }

  private _buildImpulseResponse(decaySeconds: number): void {
    const sr = this.ctx.sampleRate;
    const len = Math.floor(sr * Math.max(0.1, decaySeconds));
    const buf = this.ctx.createBuffer(2, len, sr);
    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      for (let i = 0; i < len; i++) {
        d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 2);
      }
    }
    this.reverbConvolver.buffer = buf;
  }

  // ── Oscillator ────────────────────────────────────────────────────────

  startOscillator(type: OscillatorType, frequency: number): void {
    this.stopOscillator();
    if (this.ctx.state === 'suspended') this.ctx.resume();
    this.oscillator = this.ctx.createOscillator();
    this.oscillator.type = type;
    this.oscillator.frequency.value = frequency;
    this.oscillator.connect(this.inputGain);
    this.oscillator.start();
    this._oscActive = true;
  }

  stopOscillator(): void {
    if (this.oscillator) {
      try { this.oscillator.stop(); } catch (_) { /* already stopped */ }
      this.oscillator.disconnect();
      this.oscillator = null;
    }
    this._oscActive = false;
  }

  setOscillatorFrequency(freq: number): void {
    this.oscillator?.frequency.setTargetAtTime(
      Math.max(20, Math.min(20000, freq)),
      this.ctx.currentTime, 0.005
    );
  }

  setOscillatorType(type: OscillatorType): void {
    if (this.oscillator) this.oscillator.type = type;
  }

  get isOscActive(): boolean { return this._oscActive; }

  // ── Microphone ────────────────────────────────────────────────────────

  async enableMic(): Promise<void> {
    if (this._micActive) return;
    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false,
    });
    this.micSource = this.ctx.createMediaStreamSource(this.micStream);
    this.micSource.connect(this.inputGain);
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    this._micActive = true;
  }

  disableMic(): void {
    this.micSource?.disconnect();
    this.micSource = null;
    this.micStream?.getTracks().forEach(t => t.stop());
    this.micStream = null;
    this._micActive = false;
  }

  get isMicActive(): boolean { return this._micActive; }

  // ── Effect parameters ──────────────────────────────────────────────────

  setReverb(wet: number): void {
    this.params.reverb = wet;
    this.reverbWetGain.gain.setTargetAtTime(wet, this.ctx.currentTime, 0.05);
    this.reverbDryGain.gain.setTargetAtTime(1 - wet, this.ctx.currentTime, 0.05);
  }

  setReverbDecay(secs: number): void {
    this.params.reverbDecay = secs;
    this._buildImpulseResponse(secs);
  }

  setDelay(wet: number): void {
    this.params.delay = wet;
    this.delayWetGain.gain.setTargetAtTime(wet, this.ctx.currentTime, 0.05);
    this.delayDryGain.gain.setTargetAtTime(1 - wet, this.ctx.currentTime, 0.05);
  }

  setDelayTime(secs: number): void {
    this.params.delayTime = secs;
    this.delayNode.delayTime.setTargetAtTime(secs, this.ctx.currentTime, 0.05);
  }

  setDelayFeedback(fb: number): void {
    this.params.delayFeedback = Math.min(0.95, fb);
    this.delayFeedbackGain.gain.setTargetAtTime(
      Math.min(0.95, fb), this.ctx.currentTime, 0.05
    );
  }

  setDistortion(amount: number): void {
    this.params.distortion = amount;
    this._buildDistortionCurve(amount);
  }

  setCompThreshold(db: number): void {
    this.params.compThreshold = db;
    this.compressor.threshold.setTargetAtTime(db, this.ctx.currentTime, 0.01);
  }

  setCompRatio(ratio: number): void {
    this.params.compRatio = ratio;
    this.compressor.ratio.setTargetAtTime(ratio, this.ctx.currentTime, 0.01);
  }

  setMasterGain(gain: number): void {
    this.params.masterGain = gain;
    this.masterGainNode.gain.setTargetAtTime(gain, this.ctx.currentTime, 0.05);
  }

  // ── Metering ──────────────────────────────────────────────────────────

  getSpectrumData(out: Float32Array): void {
    this.spectrumAnalyser.getFloatFrequencyData(out);
  }

  getWaveformData(out: Float32Array): void {
    this.waveformAnalyser.getFloatTimeDomainData(out);
  }

  get compressorReduction(): number {
    return this.compressor.reduction;
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────

  resume(): Promise<void> {
    return this.ctx.resume();
  }

  destroy(): void {
    this.stopOscillator();
    this.disableMic();
    this.ctx.close();
  }
}
