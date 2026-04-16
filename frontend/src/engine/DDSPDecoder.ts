import * as ort from 'onnxruntime-web';
import { IntentFrame } from './types';
import { HarmonicSynth } from './HarmonicSynth';

/**
 * DDSPDecoder
 * 
 * Uses ONNX Runtime Web to infer high-fidelity audio parameters from
 * IntentFrames [F0, Loudness, MelBands (Z)]. 
 * The inferred parameters are sent to a multi-oscillator harmonic synth
 * and a filtered noise generator (standard DDSP architecture).
 */
export class DDSPDecoder {
  private session: ort.InferenceSession | null = null;
  private synth: HarmonicSynth;
  private isModelLoaded = false;
  private isProcessing = false;

  private gateOverrideMultiplier = 1.0;
  private noiseMultiplier = 1.0;

  setGateOverride(multiplier: number) {
    this.gateOverrideMultiplier = multiplier;
  }

  setNoiseMultiplier(multiplier: number) {
    this.noiseMultiplier = multiplier;
  }

  constructor(private audioCtx: AudioContext) {
    // Connect directly to the audio context destination so audio is heard
    this.synth = new HarmonicSynth(audioCtx, audioCtx.destination);

    // Set execution providers. WebNN is highly experimental but preferred for sub-10ms.
    // Fallback to WASM or WebGL.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    
    // Explicitly point to the JSDelivr CDN matched exactly to our installed 1.24.3 binaries.
    // This perfectly bypasses all Vite Dev Server /public URL blockages, Node.js ESM strict 
    // export bindings, and local caching issues by serving raw, unbundled WASM and MJS wrappers.
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
  }

  async loadModel(url: string = '/models/ddsp_model.onnx') {
    try {
      // Prioritize WebNN for hardware acceleration, fallback to WASM
      this.session = await ort.InferenceSession.create(url, { 
        executionProviders: ['webnn', 'wasm'] 
      });
      // Also load the default Studio A Reverb Impulse Response (IR)
      const baseUrl = url.substring(0, url.lastIndexOf('/'));
      await this.synth.loadReverb(`${baseUrl}/irs/Studio_A.wav`);
      
      this.isModelLoaded = true;
      console.log(`[DDSP] Model loaded successfully from ${url} using execution providers: webnn, wasm`);
    } catch (e) {
      console.warn(`[DDSP] Failed to load ONNX model from ${url}. It might not exist yet. Run in basic additive synth mode.`);
      this.isModelLoaded = false;
    }
  }

  async setEnvironment(irName: string) {
    if (!this.synth) return;
    try {
      await this.synth.loadReverb(`/models/irs/${irName}.wav`);
      console.log(`[DDSP] Environment changed to ${irName}`);
    } catch (e) {
      console.error(`[DDSP] Failed to load environment ${irName}:`, e);
    }
  }

  getOutputNode(): AudioNode {
    return (this.synth as any).targetOut || (this.synth as any).masterGain || this.audioCtx.destination;
  }

  /**
   * Process a single IntentFrame.
   * If the model isn't loaded, maps frame data directly to the synth.
   */
  async processFrame(frame: IntentFrame) {
    // Basic fallback if no DDSP model exists
    if (!this.isModelLoaded || !this.session) {
      this.synth.update(frame);
      // Optional: manipulate synth harmonics roughly using MFCC here if desired
      return;
    }

    if (this.isProcessing) return; // Drop frame if decoder is overwhelmed (latency > 8ms)
    this.isProcessing = true;

    try {
      // 1. Prepare tensors [1, 1, 1] for f0/loudness, [1, 1, 64] for z
      const f0Tensor = new ort.Tensor('float32', Float32Array.from([frame.f0Hz]), [1, 1, 1]);
      const loudTensor = new ort.Tensor('float32', Float32Array.from([frame.rmsEnergy]), [1, 1, 1]);
      const zTensor = new ort.Tensor('float32', Float32Array.from(frame.melBands || new Float32Array(64)), [1, 1, 64]);

      // 2. Infer
      const results = await this.session.run({
        f0: f0Tensor,
        loudness: loudTensor,
        z: zTensor
      });

      // 3. Extract parameter arrays expected by a DDSP synth (harmonics & noise mags)
      // Shapes are assumed to be [1, 1, 60] for harmonics, [1, 1, 65] for noise
      const harmonicDistribution = results.harmonics?.data as Float32Array;
      const noiseMagnitudes = results.noise?.data as Float32Array;
      const reverbMix = results.reverb_mix?.data as Float32Array;
      const f0Residual = results.f0_residual?.data as Float32Array;
      const voicedMask = results.voiced_mask?.data as Float32Array;

      // Apply Feedback Controller Mutators
      if (voicedMask && this.gateOverrideMultiplier !== 1.0) {
        for (let i = 0; i < voicedMask.length; i++) {
          voicedMask[i] *= this.gateOverrideMultiplier;
        }
      }

      if (noiseMagnitudes && this.noiseMultiplier !== 1.0) {
        for (let i = 0; i < noiseMagnitudes.length; i++) {
          noiseMagnitudes[i] *= this.noiseMultiplier;
        }
      }

      // 4. Apply to synth
      if (harmonicDistribution) {
        (this.synth as any).setTarget(
          frame.f0Hz, frame.rmsEnergy, harmonicDistribution, noiseMagnitudes, reverbMix, f0Residual, voicedMask
        );
      } else {
        (this.synth as any).update(frame);
      }

    } catch (e) {
      console.error("[DDSP] Inference error:", e);
    } finally {
      this.isProcessing = false;
    }
  }

  setMuted(muted: boolean) {
    if (this.synth && typeof (this.synth as any).setMuted === 'function') {
      (this.synth as any).setMuted(muted);
    }
  }

  destroy() {
    this.synth.destroy();
  }
}
