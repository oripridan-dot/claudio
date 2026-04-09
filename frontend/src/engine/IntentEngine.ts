/**
 * IntentEngine.ts — Client-side Intent Capture and WebSocket Streaming
 *
 * Captures audio from the microphone via Web Audio API, extracts basic
 * intent features (F0 via autocorrelation, RMS loudness, spectral centroid),
 * and streams them as binary packets over WebSocket to the collaboration server.
 *
 * Also receives intent packets from remote peers and regenerates audio
 * using a simple additive synthesizer (Web Audio oscillators).
 */

// ─── Types ──────────────────────────────────────────────────────────────────

export interface IntentFrame {
  timestamp: number;
  f0Hz: number;
  confidence: number;
  loudnessDb: number;
  loudnessNorm: number;
  spectralCentroid: number;
  isOnset: boolean;
  onsetStrength: number;
  rmsEnergy: number;
}

export interface PeerInfo {
  peer_id: string;
  display_name: string;
  role: string;
  instrument: string;
  packets_sent: number;
  latency_ms: number;
}

export interface CollabMetrics {
  peer_count: number;
  total_packets: number;
  bytes_transmitted: number;
  avg_latency_ms: number;
  uptime_seconds: number;
  bandwidth_kbps: number;
}

export type IntentCallback = (frame: IntentFrame) => void;
export type PeerCallback = (peers: PeerInfo[]) => void;
export type MetricsCallback = (metrics: CollabMetrics) => void;

// ─── Intent Encoder (runs in ScriptProcessor/AudioWorklet) ──────────────────

const FRAME_RATE = 120; // Hz — 120Hz balances browser perf with decoder smoothness
const DETECT_SILENCE_THRESHOLD = 0.005;

function autocorrelationF0(
  buffer: Float32Array,
  sampleRate: number,
): { f0: number; confidence: number } {
  const n = buffer.length;
  const minPeriod = Math.floor(sampleRate / 4186); // C8
  const maxPeriod = Math.min(Math.floor(sampleRate / 33), n - 1); // C1

  if (maxPeriod <= minPeriod) return { f0: 0, confidence: 0 };

  // Normalized autocorrelation
  let bestCorr = 0;
  let bestPeriod = 0;
  let energy = 0;

  for (let i = 0; i < n; i++) energy += buffer[i] * buffer[i];
  if (energy < 1e-8) return { f0: 0, confidence: 0 };

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

function computeRMS(buffer: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
  return Math.sqrt(sum / buffer.length);
}

function computeSpectralCentroid(fft: Float32Array, sampleRate: number): number {
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

// ─── Binary Packet Serialization ────────────────────────────────────────────

function encodeIntentPacket(frame: IntentFrame, seq: number): ArrayBuffer {
  // Header: seq(u32) + ts(f32) + flags(u8) = 9 bytes
  // Payload: f0(f32) + conf(f32) + loudDb(f32) + loudNorm(f32) + centroid(f32) + onset(u8) + onsetStr(f32) = 25 bytes
  // Total: 34 bytes per packet
  const buf = new ArrayBuffer(34);
  const view = new DataView(buf);
  let offset = 0;

  // Header
  view.setUint32(offset, seq, true);
  offset += 4;
  view.setFloat32(offset, frame.timestamp, true);
  offset += 4;
  const flags =
    frame.loudnessNorm < 0.01 ? 0x08 : // SILENCE
    frame.isOnset ? 0x05 : // FULL + ONSET
    0x01; // FULL
  view.setUint8(offset, flags);
  offset += 1;

  // Payload
  view.setFloat32(offset, frame.f0Hz, true); offset += 4;
  view.setFloat32(offset, frame.confidence, true); offset += 4;
  view.setFloat32(offset, frame.loudnessDb, true); offset += 4;
  view.setFloat32(offset, frame.loudnessNorm, true); offset += 4;
  view.setFloat32(offset, frame.spectralCentroid, true); offset += 4;
  view.setUint8(offset, frame.isOnset ? 1 : 0); offset += 1;
  view.setFloat32(offset, frame.onsetStrength, true);

  return buf;
}

function decodeIntentPacket(data: ArrayBuffer): IntentFrame | null {
  if (data.byteLength < 9) return null;
  const view = new DataView(data);
  let offset = 0;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _seq = view.getUint32(offset, true); offset += 4;
  const ts = view.getFloat32(offset, true); offset += 4;
  const flags = view.getUint8(offset); offset += 1;

  if (flags & 0x08) {
    // Silence packet
    return {
      timestamp: ts, f0Hz: 0, confidence: 0,
      loudnessDb: -80, loudnessNorm: 0,
      spectralCentroid: 0, isOnset: false, onsetStrength: 0, rmsEnergy: 0,
    };
  }

  if (data.byteLength < 34) return null;

  return {
    timestamp: ts,
    f0Hz: view.getFloat32(offset, true),
    confidence: (offset += 4, view.getFloat32(offset, true)),
    loudnessDb: (offset += 4, view.getFloat32(offset, true)),
    loudnessNorm: (offset += 4, view.getFloat32(offset, true)),
    spectralCentroid: (offset += 4, view.getFloat32(offset, true)),
    isOnset: (offset += 4, view.getUint8(offset) === 1),
    onsetStrength: (offset += 1, view.getFloat32(offset, true)),
    rmsEnergy: 0,
  };
}


// ─── Intent Engine ──────────────────────────────────────────────────────────

export class IntentEngine {
  private audioCtx: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private analyser: AnalyserNode | null = null;
  private ws: WebSocket | null = null;

  private isCapturing = false;
  private captureInterval: number | null = null;
  private seq = 0;

  // Callbacks
  onLocalIntent: IntentCallback | null = null;
  onRemoteIntent: IntentCallback | null = null;
  onPeersUpdated: PeerCallback | null = null;
  onMetrics: MetricsCallback | null = null;
  onConnectionChange: ((connected: boolean) => void) | null = null;

  // State
  peerId = '';
  roomId = '';
  peers: PeerInfo[] = [];
  connected = false;

  // Remote regeneration
  private remoteOsc: OscillatorNode | null = null;
  private remoteGain: GainNode | null = null;

  // Onset detection state
  private prevRMS = 0;

  // Reconnection state
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimer: number | null = null;
  private serverUrl = '';
  private displayName = '';

  // Latency tracking
  private lastPingTs = 0;
  latencyMs = 0;

  async startCapture(): Promise<void> {
    this.audioCtx = new AudioContext({ sampleRate: 44100 });
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });

    const source = this.audioCtx.createMediaStreamSource(this.mediaStream);
    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 2048;
    source.connect(this.analyser);

    // Set up remote playback chain
    this.remoteOsc = this.audioCtx.createOscillator();
    this.remoteOsc.type = 'sine';
    this.remoteOsc.frequency.value = 0;
    this.remoteGain = this.audioCtx.createGain();
    this.remoteGain.gain.value = 0;
    this.remoteOsc.connect(this.remoteGain);
    this.remoteGain.connect(this.audioCtx.destination);
    this.remoteOsc.start();

    this.isCapturing = true;

    // Extract intent at FRAME_RATE Hz
    this.captureInterval = window.setInterval(() => {
      if (!this.analyser || !this.audioCtx) return;

      const timeDomain = new Float32Array(this.analyser.fftSize);
      this.analyser.getFloatTimeDomainData(timeDomain);

      const freqData = new Float32Array(this.analyser.frequencyBinCount);
      this.analyser.getFloatFrequencyData(freqData);

      const rms = computeRMS(timeDomain);
      const loudnessDb = 20 * Math.log10(rms + 1e-10);
      const loudnessNorm = Math.min(1, Math.max(0, (loudnessDb + 80) / 80));

      const { f0, confidence } = autocorrelationF0(timeDomain, this.audioCtx!.sampleRate);
      const centroid = computeSpectralCentroid(freqData, this.audioCtx!.sampleRate);

      // Simple onset detection via RMS jump
      const rmsJump = rms - this.prevRMS;
      const isOnset = rmsJump > 0.05 && rms > 0.02;
      this.prevRMS = rms;

      const frame: IntentFrame = {
        timestamp: performance.now(),
        f0Hz: f0,
        confidence,
        loudnessDb,
        loudnessNorm,
        spectralCentroid: centroid,
        isOnset,
        onsetStrength: Math.max(0, rmsJump),
        rmsEnergy: rms,
      };

      this.onLocalIntent?.(frame);

      // Send via WebSocket
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.seq++;
        const packet = encodeIntentPacket(frame, this.seq);
        this.ws.send(packet);
      }
    }, Math.floor(1000 / FRAME_RATE));
  }

  stopCapture(): void {
    this.isCapturing = false;
    if (this.captureInterval !== null) {
      clearInterval(this.captureInterval);
      this.captureInterval = null;
    }
    this.mediaStream?.getTracks().forEach(t => t.stop());
    this.remoteOsc?.stop();
    this.audioCtx?.close();
    this.audioCtx = null;
  }

  async connectToRoom(serverUrl: string, roomId: string, name: string): Promise<void> {
    this.roomId = roomId;
    this.serverUrl = serverUrl;
    this.displayName = name;
    this.reconnectAttempts = 0;
    this._connectWebSocket();
  }

  private _connectWebSocket(): void {
    const wsUrl = `${this.serverUrl.replace('http', 'ws')}/ws/collab/${this.roomId}?name=${encodeURIComponent(this.displayName)}`;
    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.connected = true;
      this.reconnectAttempts = 0;
      this.onConnectionChange?.(true);

      // Start ping interval for latency measurement
      this._startPing();
    };

    this.ws.onclose = () => {
      this.connected = false;
      this.onConnectionChange?.(false);
      this._attemptReconnect();
    };

    this.ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        // Binary: remote intent packet
        const frame = decodeIntentPacket(ev.data);
        if (frame) {
          this.onRemoteIntent?.(frame);
          this.regenerateFromIntent(frame);
        }
      } else {
        // JSON: signaling
        const msg = JSON.parse(ev.data);
        switch (msg.type) {
          case 'welcome':
            this.peerId = msg.peer_id;
            this.peers = msg.peers || [];
            this.onPeersUpdated?.(this.peers);
            break;
          case 'peer_joined':
          case 'peer_updated':
          case 'peer_left':
            this.peers = msg.peers || [];
            this.onPeersUpdated?.(this.peers);
            break;
          case 'metrics':
            this.onMetrics?.(msg);
            break;
          case 'pong': {
            // Calculate round-trip latency
            if (this.lastPingTs > 0) {
              this.latencyMs = performance.now() - this.lastPingTs;
            }
            break;
          }
        }
      }
    };
  }

  private _startPing(): void {
    // Send ping every 3 seconds for latency measurement
    const pingInterval = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.lastPingTs = performance.now();
        this.ws.send(JSON.stringify({ type: 'ping' }));
      } else {
        clearInterval(pingInterval);
      }
    }, 3000);
  }

  private _attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    if (this.reconnectTimer !== null) return;

    this.reconnectAttempts++;
    const delay = Math.min(1000 * 2 ** (this.reconnectAttempts - 1), 30000);

    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this._connectWebSocket();
    }, delay);
  }

  disconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent auto-reconnect
    this.ws?.close();
    this.ws = null;
    this.connected = false;
  }

  requestMetrics(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'metrics_request' }));
    }
  }

  setInstrument(instrument: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'instrument_set', instrument }));
    }
  }

  private regenerateFromIntent(frame: IntentFrame): void {
    if (!this.remoteOsc || !this.remoteGain || !this.audioCtx) return;

    // Set oscillator frequency if voiced
    if (frame.f0Hz > 0 && frame.confidence > 0.3) {
      this.remoteOsc.frequency.setTargetAtTime(
        frame.f0Hz, this.audioCtx.currentTime, 0.01,
      );
    }

    // Set volume from loudness
    this.remoteGain.gain.setTargetAtTime(
      frame.loudnessNorm * 0.3, this.audioCtx.currentTime, 0.01,
    );
  }
}
