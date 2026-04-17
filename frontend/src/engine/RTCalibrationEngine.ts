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
}

export class RTCalibrationEngine {
  private audioCtx: AudioContext;
  
  // Analysers
  private inputAnalyser: AnalyserNode;
  private outputAnalyser: AnalyserNode;

  private inputConnected = false;
  private outputConnected = false;

  public onAutoCalibrate?: (metrics: CalibrationMetrics) => void;

  public params: CalibrationParams = {
    fftSize: 2048,
    gainCompStrength: 0.5,
    eqCompStrength: 0.5,
    smoothingMs: 50.0
  };

  public metrics: CalibrationMetrics = {
    coherence: 0,
    inputMagnitudeDb: -80,
    outputMagnitudeDb: -80,
    magnitudeDeltaDb: 0,
    inputCentroidHz: 0,
    outputCentroidHz: 0,
    freqDeltaHz: 0
  };

  constructor(audioCtx: AudioContext) {
    this.audioCtx = audioCtx;
    this.inputAnalyser = this.audioCtx.createAnalyser();
    this.outputAnalyser = this.audioCtx.createAnalyser();
    
    this.inputAnalyser.fftSize = this.params.fftSize;
    this.outputAnalyser.fftSize = this.params.fftSize;
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


  
  /** Also need a way to tap the raw synth to measure it */
  connectOutputTap(sourceNode: AudioNode) {
    sourceNode.connect(this.outputAnalyser);
    this.outputConnected = true;
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

    if (this.onAutoCalibrate) {
        this.onAutoCalibrate(this.metrics);
    }

    return this.metrics;
  }

  destroy() {
      // Clean up analyzers
      this.inputAnalyser.disconnect();
      this.outputAnalyser.disconnect();
  }
}
