/**
 * ResynthEngine.ts — Neural Audio Codec Streaming (EnCodec)
 *
 * Streams raw PCM chunks to the server's /ws/audio endpoint (EnCodec neural codec),
 * receives compressed audio, and schedules it for glitch-free playback.
 *
 * Audio quality: Near-transparent at 6 kbps (EnCodec, Meta Research).
 * Latency: ~80–150ms round-trip (Cloud Run + codec). Good for capture-playback.
 */

const CHUNK_SAMPLES = 2048;  // ≈42ms at 48000Hz (halved for lower latency UI responsiveness)
const SAMPLE_RATE   = 48000; // Native WebRTC rate to prevent aliasing/resampling
const FADE_SAMPLES  = 64;    // linear fade-in/out to prevent inter-chunk clicks

export type ResynthStateCallback = (state: 'idle' | 'capturing' | 'playing' | 'error') => void;

export class ResynthEngine {
  private audioCtx: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private ws: WebSocket | null = null;

  private buffer: Float32Array = new Float32Array(0);
  private processor: ScriptProcessorNode | null = null;
  private nextPlayTime = 0;

  onStateChange: ResynthStateCallback | null = null;
  latencyMs = 0;
  private _pingTs = 0;

  // Added for RT Calibration
  masterOut: GainNode | null = null;
  private inputSource: MediaStreamAudioSourceNode | null = null;

  getAudioContext(): AudioContext | null { return this.audioCtx; }
  getInputSource(): AudioNode | null { return this.inputSource; }

  redirectOutput(target: AudioNode) {
    if (this.masterOut && this.audioCtx) {
      this.masterOut.disconnect();
      this.masterOut.connect(target);
    }
  }

  /** Connect to the NeuralCodec (EnCodec) WebSocket and start streaming. */
  async start(serverUrl: string): Promise<void> {
    this.audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: 'interactive' });
    this.masterOut = this.audioCtx.createGain();
    this.masterOut.connect(this.audioCtx.destination);
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { 
        echoCancellation: false, 
        noiseSuppression: false, 
        autoGainControl: false,
        channelCount: 2,
        sampleRate: SAMPLE_RATE
      },
    });
    if (this.audioCtx.state === 'suspended') {
      await this.audioCtx.resume();
    }
    this.inputSource = this.audioCtx.createMediaStreamSource(this.mediaStream);

    const wsUrl = serverUrl.replace(/^http/, 'ws') + '/ws/audio';
    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this._startCapture();
      this.onStateChange?.('capturing');
    };

    this.ws.onmessage = (ev) => {
      if (!(ev.data instanceof ArrayBuffer)) return;
      const pcm = new Float32Array(ev.data);
      this._scheduleChunk(pcm);
      // Measure round-trip latency
      if (this._pingTs > 0) {
        this.latencyMs = performance.now() - this._pingTs;
        this._pingTs = 0;
      }
    };

    this.ws.onerror = () => this.onStateChange?.('error');
    this.ws.onclose = () => {
      if (this.processor) this.onStateChange?.('idle');
    };
  }

  stop(): void {
    this.processor?.disconnect();
    this.processor = null;
    this.mediaStream?.getTracks().forEach(t => t.stop());
    this.ws?.close();
    this.audioCtx?.close();
    this.audioCtx = null;
    this.buffer = new Float32Array(0);
    this.nextPlayTime = 0;
    this.onStateChange?.('idle');
  }

  private _startCapture(): void {
    if (!this.audioCtx || !this.mediaStream || !this.inputSource) return;

    // ScriptProcessorNode: simple, synchronous, no worklet overhead for this use case
    this.processor = this.audioCtx.createScriptProcessor(CHUNK_SAMPLES, 1, 1);
    this.inputSource.connect(this.processor);

    this.processor.onaudioprocess = (ev) => {
      if (this.ws?.readyState !== WebSocket.OPEN) return;

      const input = ev.inputBuffer.getChannelData(0);
      const chunk = new Float32Array(input); // copy

      this._pingTs = performance.now();
      this.ws.send(chunk.buffer);
    };

    source.connect(this.processor);
    // Must connect to destination for onaudioprocess to fire (browser quirk)
    this.processor.connect(this.audioCtx.createGain()); // silent gain (0 default)
  }

  private _scheduleChunk(pcm: Float32Array): void {
    if (!this.audioCtx) return;

    // Apply short linear fades to prevent inter-chunk clicks
    const faded = new Float32Array(pcm);
    const f = Math.min(FADE_SAMPLES, Math.floor(faded.length / 4));
    for (let i = 0; i < f; i++) {
      faded[i] *= i / f;
      faded[faded.length - 1 - i] *= i / f;
    }

    const buf = this.audioCtx.createBuffer(1, faded.length, SAMPLE_RATE);
    buf.copyToChannel(faded, 0);

    const src = this.audioCtx.createBufferSource();
    src.buffer = buf;
    if (this.masterOut) {
      src.connect(this.masterOut);
    } else {
      src.connect(this.audioCtx.destination);
    }

    const now = this.audioCtx.currentTime;
    if (this.nextPlayTime < now + 0.01) {
      // First chunk or underrun — reset with small pre-buffer
      this.nextPlayTime = now + 0.06;
    }

    src.start(this.nextPlayTime);
    this.nextPlayTime += buf.duration;
    this.onStateChange?.('playing');
  }
}
