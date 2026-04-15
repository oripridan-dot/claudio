/**
 * RTCalibrationEngine.ts — Live signal comparison and dynamic audio compensation.
 */

export interface CalibrationParams {
  fftSize: number;           // 1024, 2048, 4096
  gainCompStrength: number;  // 0.0 to 1.0
  eqCompStrength: number;    // 0.0 to 1.0
  smoothingMs: number;       // 10.0 to 500.0
}

export interface CalibrationMetrics {
  coherence: number;
  inputMagnitudeDb: number;
  outputMagnitudeDb: number;
  magnitudeDeltaDb: number;
  inputCentroidHz: number;
  outputCentroidHz: number;
  freqDeltaHz: number;
  compensationGainLvl: number;
  compensationFilterFreq: number;
}

export class RTCalibrationEngine {
  private audioCtx: AudioContext;
  
  // Analysers
  private inputAnalyser: AnalyserNode;
  private outputAnalyser: AnalyserNode;

  // Compensation chain applied to the output stream
  private compensationGain: GainNode;
  private compensationFilter: BiquadFilterNode;

  // Parameters
  params: CalibrationParams = {
    fftSize: 2048,
    gainCompStrength: 0.5,
    eqCompStrength: 0.5,
    smoothingMs: 50
  };

  // Metrics
  metrics: CalibrationMetrics = {
    coherence: 0,
    inputMagnitudeDb: -80,
    outputMagnitudeDb: -80,
    magnitudeDeltaDb: 0,
    inputCentroidHz: 0,
    outputCentroidHz: 0,
    freqDeltaHz: 0,
    compensationGainLvl: 1.0,
    compensationFilterFreq: 1000
  };

  // State
  private inputConnected = false;
  private outputConnected = false;

  constructor(audioCtx: AudioContext) {
    this.audioCtx = audioCtx;
    this.inputAnalyser = this.audioCtx.createAnalyser();
    this.outputAnalyser = this.audioCtx.createAnalyser();
    
    this.inputAnalyser.fftSize = this.params.fftSize;
    this.outputAnalyser.fftSize = this.params.fftSize;
    
    this.compensationGain = this.audioCtx.createGain();
    this.compensationFilter = this.audioCtx.createBiquadFilter();
    
    // Default simple High-Shelf/Low-Shelf combo filter could be modeled,
    // but we'll use a broad Peaking filter to adjust spectral balance dynamically
    this.compensationFilter.type = 'peaking';
    this.compensationFilter.Q.value = 0.5; // broad Q
    this.compensationFilter.frequency.value = 1000;
    this.compensationFilter.gain.value = 0; // neutral
    
    // Connect output chain: Audio in -> Filter -> Gain -> (User connects to destination)
    this.compensationFilter.connect(this.compensationGain);
    
    // The output analyser reads *before* or *after* compensation?
    // Let's read *before* compensation to see the raw delta and compute compensation accurately.
    // Actually, reading AFTER compensation might cause a feedback loop in the math unless handled right.
    // We will assume `connectTargetSource` attaches the raw synth, which feeds into `compensationFilter`.
  }

  setParams(newParams: Partial<CalibrationParams>) {
    Object.assign(this.params, newParams);
    
    if (newParams.fftSize && newParams.fftSize !== this.inputAnalyser.fftSize) {
      // Must be power of 2
      const valid = [256, 512, 1024, 2048, 4096, 8192];
      if (valid.includes(newParams.fftSize)) {
        this.inputAnalyser.fftSize = newParams.fftSize;
        this.outputAnalyser.fftSize = newParams.fftSize;
      }
    }
  }

  /** Wire the raw microphone into the input analyser */
  connectInputSource(sourceNode: AudioNode) {
    // Avoid double-connecting if already connected to something? 
    // Usually safe to just connect.
    sourceNode.connect(this.inputAnalyser);
    this.inputConnected = true;
  }

  /** 
   * Wire the synthesized output to pass through our compensation chain.
   * Returns the node that the generator should connect TO.
   */
  getCompensationInputNode(): AudioNode {
    return this.compensationFilter;
  }
  
  /** Also need a way to tap the raw synth to measure it */
  connectOutputTap(sourceNode: AudioNode) {
    sourceNode.connect(this.outputAnalyser);
    this.outputConnected = true;
  }

  /** The final node to connect to the AudioContext destination */
  getFinalOutputNode(): AudioNode {
    return this.compensationGain;
  }

  private computeRMS(data: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    return Math.sqrt(sum / data.length);
  }

  private computeSpectralCentroid(freqData: Float32Array, sampleRate: number): number {
    let weightedSum = 0;
    let totalMag = 0;
    const binWidth = sampleRate / (freqData.length * 2);
    for (let i = 0; i < freqData.length; i++) {
        // convert dB to linear magnitude roughly
      const mag = Math.pow(10, freqData[i] / 20);
      weightedSum += mag * (i * binWidth);
      totalMag += mag;
    }
    return totalMag > 0 ? weightedSum / totalMag : 0;
  }

  private computeCoherence(inpTime: Float32Array, outTime: Float32Array): number {
    // Simplified coherence: normalized cross-correlation at lag 0 (since we assume aligned or broadly matched envelopes)
    let dot = 0;
    let normInp = 0;
    let normOut = 0;
    for (let i = 0; i < inpTime.length; i++) {
      dot += Math.abs(inpTime[i] * outTime[i]);
      normInp += inpTime[i] * inpTime[i];
      normOut += outTime[i] * outTime[i];
    }
    if (normInp === 0 || normOut === 0) return 0;
    return dot / Math.sqrt(normInp * normOut);
  }

  /** Call this continuously via requestAnimationFrame from the UI */
  updateMetrics() {
    if (!this.inputConnected || !this.outputConnected) return this.metrics;

    const inpTime = new Float32Array(this.inputAnalyser.fftSize);
    const outTime = new Float32Array(this.outputAnalyser.fftSize);
    this.inputAnalyser.getFloatTimeDomainData(inpTime);
    this.outputAnalyser.getFloatTimeDomainData(outTime);

    const inFreq = new Float32Array(this.inputAnalyser.frequencyBinCount);
    const outFreq = new Float32Array(this.outputAnalyser.frequencyBinCount);
    this.inputAnalyser.getFloatFrequencyData(inFreq);
    this.outputAnalyser.getFloatFrequencyData(outFreq);

    const inRms = this.computeRMS(inpTime);
    const outRms = this.computeRMS(outTime);
    
    const inDb = 20 * Math.log10(inRms + 1e-10);
    const outDb = 20 * Math.log10(outRms + 1e-10);
    const magDelta = outDb - inDb;

    const inCent = this.computeSpectralCentroid(inFreq, this.audioCtx.sampleRate);
    const outCent = this.computeSpectralCentroid(outFreq, this.audioCtx.sampleRate);
    const freqDelta = outCent - inCent;

    const coherence = this.computeCoherence(inpTime, outTime);

    this.metrics.inputMagnitudeDb = inDb;
    this.metrics.outputMagnitudeDb = outDb;
    this.metrics.magnitudeDeltaDb = magDelta;
    this.metrics.inputCentroidHz = inCent;
    this.metrics.outputCentroidHz = outCent;
    this.metrics.freqDeltaHz = freqDelta;
    this.metrics.coherence = coherence;

    this.applyCompensation(inDb, outDb, inCent, outCent);

    return this.metrics;
  }

  private applyCompensation(inDb: number, outDb: number, inCent: number, outCent: number) {
    const now = this.audioCtx.currentTime;
    const smoothSec = this.params.smoothingMs / 1000;

    // 1. Gain Compensation
    // Only compensate if there is active signal (e.g. > -60dB)
    if (inDb > -60 && this.params.gainCompStrength > 0) {
      // We want outDb + correction = inDb => correction_dB = inDb - outDb
      const targetGainDb = (inDb - outDb) * this.params.gainCompStrength;
      // Clamp reasonable limits (-24 to +24 dB)
      const clampedDb = Math.max(-24, Math.min(24, targetGainDb));
      const targetLinearGain = Math.pow(10, clampedDb / 20);
      
      this.compensationGain.gain.setTargetAtTime(targetLinearGain, now, smoothSec);
      this.metrics.compensationGainLvl = targetLinearGain;
    } else {
      this.compensationGain.gain.setTargetAtTime(1.0, now, smoothSec);
      this.metrics.compensationGainLvl = 1.0;
    }

    // 2. EQ Compensation (Spectral Centroid Match)
    // If output centroid is lower than input, we need higher frequency boost.
    if (inDb > -60 && this.params.eqCompStrength > 0) {
      // Find proportional frequency difference
      const freqRatio = inCent / (outCent + 1e-5);
      
      // If freqRatio > 1, input is brighter. We boost the highs via peaking filter at high freq.
      // If freqRatio < 1, output is too bright. We cut the highs.
      
      // We arbitrarily target the filter at 3000 Hz to act broadly on brightness
      this.compensationFilter.frequency.setTargetAtTime(3000, now, smoothSec);
      this.metrics.compensationFilterFreq = 3000;

      // Filter gain logic: 
      let targetEqGainDb = Math.log2(freqRatio) * 6.0 * this.params.eqCompStrength; 
      targetEqGainDb = Math.max(-12, Math.min(12, targetEqGainDb)); // +/- 12dB clamp
      
      this.compensationFilter.gain.setTargetAtTime(targetEqGainDb, now, smoothSec);
    } else {
      this.compensationFilter.gain.setTargetAtTime(0, now, smoothSec);
    }
  }

  destroy() {
    this.compensationFilter.disconnect();
    this.compensationGain.disconnect();
  }
}
