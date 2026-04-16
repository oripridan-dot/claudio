import * as ort from 'onnxruntime-web';
import { IntentFrame } from './types';
import { HarmonicSynth } from './HarmonicSynth';

/**
 * DDSPDecoder
 * 
 * Uses ONNX Runtime Web to infer high-fidelity audio parameters from
 * IntentFrames [F0, Loudness, MFCC (Z)]. 
 * The inferred parameters are sent to a multi-oscillator harmonic synth
 * and a filtered noise generator (standard DDSP architecture).
 */
export class DDSPDecoder {
  private session: ort.InferenceSession | null = null;
  private synth: HarmonicSynth;
  private isModelLoaded = false;
  private isProcessing = false;

  constructor(private audioCtx: AudioContext) {
    const dummyDest = audioCtx.createGain(); // Create a generic output node
    this.synth = new HarmonicSynth(audioCtx, dummyDest); 
    
    // Set execution providers. WebNN is highly experimental but preferred for sub-10ms.
    // Fallback to WASM or WebGL.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
  }

  async loadModel(url: string = '/models/ddsp_model.onnx') {
    try {
      // Prioritize WebNN for hardware acceleration, fallback to WASM
      this.session = await ort.InferenceSession.create(url, { 
        executionProviders: ['webnn', 'wasm'] 
      });
      this.isModelLoaded = true;
      console.log(`[DDSP] Model loaded successfully from ${url} using execution providers: webnn, wasm`);
    } catch (e) {
      console.warn(`[DDSP] Failed to load ONNX model from ${url}. It might not exist yet. Run in basic additive synth mode.`);
      this.isModelLoaded = false;
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
      this.synth.setTarget(frame.f0Hz, frame.rmsEnergy);
      // Optional: manipulate synth harmonics roughly using MFCC here if desired
      return;
    }

    if (this.isProcessing) return; // Drop frame if decoder is overwhelmed (latency > 8ms)
    this.isProcessing = true;

    try {
      // 1. Prepare tensors [1, 1, 1] for f0/loudness, [1, 1, 13] for mfcc
      const f0Tensor = new ort.Tensor('float32', Float32Array.from([frame.f0Hz]), [1, 1, 1]);
      const loudTensor = new ort.Tensor('float32', Float32Array.from([frame.rmsEnergy]), [1, 1, 1]);
      const zTensor = new ort.Tensor('float32', Float32Array.from(frame.mfcc || new Array(13).fill(0)), [1, 1, 13]);

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

      // 4. Apply to synth
      if (harmonicDistribution) {
        (this.synth as any).setTarget(frame.f0Hz, frame.rmsEnergy, harmonicDistribution);
      } else {
        (this.synth as any).update(frame);
      }
      
      // Note: A true DDSP implementation would also route noiseMagnitudes to a 
      // dynamically filtered noise generator here.

    } catch (e) {
      console.error("[DDSP] Inference error:", e);
    } finally {
      this.isProcessing = false;
    }
  }

  destroy() {
    this.synth.stop();
  }
}
