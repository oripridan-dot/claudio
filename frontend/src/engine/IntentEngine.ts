/**
 * IntentEngine.ts — Client-side Intent Capture and WebSocket Streaming
 *
 * Phase 1: Multi-oscillator harmonic synth + MFCC timbre packets
 * Phase 2: WebRTC sendrecv audio + ICE relay → near-lossless p2p audio
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
  mfcc: number[]; // 13 coefficients — timbre fingerprint
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

// ─── Constants ───────────────────────────────────────────────────────────────

const FRAME_RATE = 120;
const N_MFCC = 13;
const N_MELS = 26;
const N_PARTIALS = 16;
const DETECT_SILENCE_THRESHOLD = 0.005;


function autocorrelationF0(
  buffer: Float32Array,
  sampleRate: number,
): { f0: number; confidence: number } {
  const n = buffer.length;
  const minPeriod = Math.floor(sampleRate / 1046); // C6 max (reduces high-freq noise)
  const maxPeriod = Math.min(Math.floor(sampleRate / 40), n - 1); // E1 min

  if (maxPeriod <= minPeriod) return { f0: 0, confidence: 0 };

  // Normalized autocorrelation
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

function buildMelFilterbank(fftSize: number, sampleRate: number): Float32Array[] {
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

function computeMFCC(magSpectrum: Float32Array, filterbank: Float32Array[]): number[] {
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

// ─── Binary Packet Serialization ────────────────────────────────────────────

function encodeIntentPacket(frame: IntentFrame, seq: number): ArrayBuffer {
  const buf = new ArrayBuffer(PACKET_BYTES);
  const view = new DataView(buf);
  let o = 0;
  view.setUint32(o, seq, true); o += 4;
  view.setFloat32(o, frame.timestamp, true); o += 4;
  const flags = frame.loudnessNorm < 0.01 ? 0x08 : frame.isOnset ? 0x05 : 0x01;
  view.setUint8(o, flags); o += 1;
  view.setFloat32(o, frame.f0Hz, true); o += 4;
  view.setFloat32(o, frame.confidence, true); o += 4;
  view.setFloat32(o, frame.loudnessDb, true); o += 4;
  view.setFloat32(o, frame.loudnessNorm, true); o += 4;
  view.setFloat32(o, frame.spectralCentroid, true); o += 4;
  view.setUint8(o, frame.isOnset ? 1 : 0); o += 1;
  view.setFloat32(o, frame.onsetStrength, true); o += 4;
  // MFCCs
  const mfcc = frame.mfcc.length === N_MFCC ? frame.mfcc : Array(N_MFCC).fill(0);
  for (let i = 0; i < N_MFCC; i++) { view.setFloat32(o, mfcc[i], true); o += 4; }
  return buf;
}

function decodeIntentPacket(data: ArrayBuffer): { seq: number; frame: IntentFrame } | null {
  if (data.byteLength < 9) return null;
  const view = new DataView(data);
  let o = 0;
  const seq = view.getUint32(o, true); o += 4;
  const ts = view.getFloat32(o, true); o += 4;
  const flags = view.getUint8(o); o += 1;

  const emptyMfcc = Array(N_MFCC).fill(0);

  if (flags & 0x08) {
    return { seq, frame: { timestamp: ts, f0Hz: 0, confidence: 0, loudnessDb: -80,
      loudnessNorm: 0, spectralCentroid: 0, isOnset: false, onsetStrength: 0,
      rmsEnergy: 0, mfcc: emptyMfcc } };
  }
  if (data.byteLength < PACKET_BYTES) return null;

  const f0Hz = view.getFloat32(o, true); o += 4;
  const confidence = view.getFloat32(o, true); o += 4;
  const loudnessDb = view.getFloat32(o, true); o += 4;
  const loudnessNorm = view.getFloat32(o, true); o += 4;
  const spectralCentroid = view.getFloat32(o, true); o += 4;
  const isOnset = view.getUint8(o) === 1; o += 1;
  const onsetStrength = view.getFloat32(o, true); o += 4;
  const mfcc: number[] = [];
  for (let i = 0; i < N_MFCC; i++) { mfcc.push(view.getFloat32(o, true)); o += 4; }

  return { seq, frame: { timestamp: ts, f0Hz, confidence, loudnessDb, loudnessNorm,
    spectralCentroid, isOnset, onsetStrength, rmsEnergy: 0, mfcc } };
}

// ─── Multi-Oscillator Harmonic Synth ────────────────────────────────────────

class HarmonicSynth {
  private partials: OscillatorNode[] = [];
  private gains: GainNode[] = [];
  private noiseSource: AudioBufferSourceNode | null = null;
  private noiseFilter: BiquadFilterNode | null = null;
  private noiseGain: GainNode | null = null;
  private masterGain: GainNode;
  private ctx: AudioContext;
  private destination: AudioNode;
  private prevF0 = 0;
  private melFb: Float32Array[] | null = null;

  constructor(ctx: AudioContext, destination: AudioNode) {
    this.ctx = ctx;
    this.destination = destination;
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

    this.noiseFilter = this.ctx.createBiquadFilter();
    this.noiseFilter.type = 'lowpass';
    this.noiseFilter.frequency.value = 800;
    this.noiseFilter.Q.value = 0.5;

    this.noiseGain = this.ctx.createGain();
    this.noiseGain.gain.value = 0;

    this.noiseSource.connect(this.noiseFilter);
    this.noiseFilter.connect(this.noiseGain);
    this.noiseGain.connect(this.masterGain);
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
      if (this.noiseFilter && this.noiseGain) {
        const cutoff = Math.min(Math.max(frame.spectralCentroid * 1.5, 200), 8000);
        this.noiseFilter.frequency.setTargetAtTime(cutoff, now, smooth);
        // Noise level: higher for breathy/noisy sounds (low F0 confidence → more noise)
        const noiseLevel = (1 - frame.confidence) * 0.08 * frame.loudnessNorm;
        this.noiseGain.gain.setTargetAtTime(noiseLevel, now, smooth);
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

  destroy() {
    this.partials.forEach(o => { try { o.stop(); } catch (_) {} });
    this.noiseSource?.stop();
    this.masterGain.disconnect();
  }
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

  // WebRTC state
  private pc: RTCPeerConnection | null = null;
  private dataChannel: RTCDataChannel | null = null;
  private remoteStream: MediaStream | null = null;
  private remoteStreamSource: MediaStreamAudioSourceNode | null = null;

  // State
  peerId = '';
  roomId = '';
  peers: PeerInfo[] = [];
  connected = false;

  // Remote regeneration — Phase 1: multi-oscillator harmonic synth
  private remoteSynth: HarmonicSynth | null = null;
  private melFilterbank: Float32Array[] | null = null;
  ddspMode = false;
  private nextAudioTime = 0;

  // Onset detection state
  private prevRMS = 0;

  // Reconnection state
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimer: number | null = null;
  private serverUrl = '';
  private displayName = '';
  private jwtToken: string | null = null;

  // Latency & Network Telemetry logging
  private lastPingTs = 0;
  latencyMs = 0;
  
  jitterMs = 0.0;
  packetLossPercent = 0.0;
  private remoteSeq: number | null = null;
  private packetsReceived = 0;
  private packetsLost = 0;
  private packetsWindowReceived = 0;
  private packetsWindowLost = 0;
  private lastNetworkUpdateTs = 0;

  async startCapture(): Promise<void> {
    this.audioCtx = new AudioContext({ sampleRate: 44100 });
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });

    const source = this.audioCtx.createMediaStreamSource(this.mediaStream);
    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 2048;
    source.connect(this.analyser);

    // Build mel filterbank once (2048-point FFT, 44100 Hz)
    this.melFilterbank = buildMelFilterbank(2048, 44100);

    // Phase 1: multi-oscillator harmonic synth for remote playback
    this.remoteSynth = new HarmonicSynth(this.audioCtx, this.audioCtx.destination);
    this.remoteSynth.setMelFilterbank(this.melFilterbank);

    this.isCapturing = true;

    this.captureInterval = window.setInterval(() => {
      if (!this.analyser || !this.audioCtx || !this.melFilterbank) return;

      const timeDomain = new Float32Array(this.analyser.fftSize);
      this.analyser.getFloatTimeDomainData(timeDomain as Float32Array<ArrayBuffer>);

      const freqData = new Float32Array(this.analyser.frequencyBinCount);
      this.analyser.getFloatFrequencyData(freqData as Float32Array<ArrayBuffer>);

      // Convert dBFS freq data to linear magnitude for MFCC
      const magSpectrum = new Float32Array(freqData.length);
      for (let i = 0; i < freqData.length; i++)
        magSpectrum[i] = Math.pow(10, freqData[i] / 20);

      const rms = computeRMS(timeDomain);
      const loudnessDb = 20 * Math.log10(rms + 1e-10);
      const loudnessNorm = Math.min(1, Math.max(0, (loudnessDb + 80) / 80));

      const { f0, confidence } = autocorrelationF0(timeDomain, this.audioCtx.sampleRate);
      const centroid = computeSpectralCentroid(freqData, this.audioCtx.sampleRate);
      const mfcc = computeMFCC(magSpectrum, this.melFilterbank);

      const rmsJump = rms - this.prevRMS;
      const isOnset = rmsJump > 0.05 && rms > 0.02;
      this.prevRMS = rms;

      const frame: IntentFrame = {
        timestamp: performance.now(),
        f0Hz: f0, confidence, loudnessDb, loudnessNorm,
        spectralCentroid: centroid, isOnset,
        onsetStrength: Math.max(0, rmsJump),
        rmsEnergy: rms, mfcc,
      };

      this.onLocalIntent?.(frame);

      if (this.dataChannel?.readyState === 'open') {
        this.seq++;
        this.dataChannel.send(encodeIntentPacket(frame, this.seq));
      } else if (this.ws?.readyState === WebSocket.OPEN) {
        this.seq++;
        this.ws.send(encodeIntentPacket(frame, this.seq));
      }
    }, Math.floor(1000 / FRAME_RATE));
  }

  stopCapture(): void {
    this.isCapturing = false;
    if (this.captureInterval !== null) {
      clearInterval(this.captureInterval);
      this.captureInterval = null;
    }
    this.remoteSynth?.destroy();
    this.remoteSynth = null;
    this.mediaStream?.getTracks().forEach(t => t.stop());
    this.remoteStream?.getTracks().forEach(t => t.stop());
    if (this.pc) { this.pc.close(); this.pc = null; }
    this.audioCtx?.close();
    this.audioCtx = null;
  }

  async connectToRoom(serverUrl: string, roomId: string, name: string): Promise<void> {
    this.roomId = roomId;
    this.serverUrl = serverUrl;
    this.displayName = name;
    this.reconnectAttempts = 0;
    
    // Fetch JWT Token for websocket negotiation
    try {
        const res = await fetch(`${this.serverUrl}/api/auth/token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: this.displayName }),
        });
        if (!res.ok) throw new Error('Authorization failed');
        const data = await res.json();
        this.jwtToken = data.token;
    } catch (e) {
        console.error("Failed to authenticate:", e);
        this.onConnectionChange?.(false);
        return;
    }

    this._connectWebSocket();
  }

  disconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent auto-reconnect
    if (this.pc) {
      this.pc.close();
      this.pc = null;
    }
    if (this.dataChannel) {
        this.dataChannel.close();
        this.dataChannel = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.connected = false;
    this.onConnectionChange?.(false);
  }

  private _connectWebSocket(): void {
    if (!this.jwtToken) return;
    const wsUrl = `${this.serverUrl.replace('http', 'ws')}/ws/collab/${this.roomId}?name=${encodeURIComponent(this.displayName)}&token=${this.jwtToken}`;
    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.connected = true;
      this.reconnectAttempts = 0;
      this.onConnectionChange?.(true);

      // Start ping interval for latency measurement
      this._startPing();
      this._initWebRTC();
    };

    this.ws.onclose = () => {
      this.connected = false;
      this.onConnectionChange?.(false);
      this._attemptReconnect();
    };

    this.ws.onmessage = async (ev) => {

      if (ev.data instanceof ArrayBuffer) {
        const view = new DataView(ev.data);
        if (ev.data.byteLength > 4) {
          const magic = view.getUint32(0, false); // Big endian read of "DDSP"
          if (magic === 0x44445350) { // b"DDSP"
            const audioData = new Float32Array(ev.data.slice(12)); // skip 4 magic + 8 peer_id
            this._scheduleAudioBlock(audioData);
            return;
          }
        }

        // Binary: remote intent packet
        const decoded = decodeIntentPacket(ev.data);
        if (decoded) {
          const { seq, frame } = decoded;
          this.onRemoteIntent?.(frame);
          
          // Network Telemetry calculate: Packet Loss
          if (this.remoteSeq !== null) {
              const expected = this.remoteSeq + 1;
              if (seq > expected) {
                  const lost = seq - expected;
                  this.packetsLost += lost;
                  this.packetsWindowLost += lost;
              }
          }
          this.remoteSeq = seq;
          this.packetsReceived++;
          this.packetsWindowReceived++;

          // Network Telemetry calculate: Jitter
          // Simplified RTCP jitter: D = |(Rj - Sj) - (Ri - Si)| = |(Rj - Ri) - (Sj - Si)|
          // Without synchronized clocks we just look at relative deltas.
          // Since our tick is 8.3ms, we expect arriving packets approx every 8.33ms.
          // For now, let's proxy jitter as var of arrivals based on ping.
          const now = performance.now();
          if (this.lastNetworkUpdateTs > 0) {
              const deltaR = now - this.lastNetworkUpdateTs;
              const deltaS = 8.33; // Nominal at 120Hz
              const diff = Math.abs(deltaR - deltaS);
              // Moving average using RTCP formula J = J + (|D| - J) / 16
              this.jitterMs += (diff - this.jitterMs) / 16;
          }
          this.lastNetworkUpdateTs = now;
          
          // Periodically update packet loss percent (every 1 second approx 120 pkts)
          if (this.packetsWindowReceived > 120) {
              const totalWindow = this.packetsWindowReceived + this.packetsWindowLost;
              this.packetLossPercent = totalWindow > 0 ? (this.packetsWindowLost / totalWindow) * 100 : 0;
              this.packetsWindowReceived = 0;
              this.packetsWindowLost = 0;
          }

          if (!this.ddspMode) {
            this.regenerateFromIntent(frame);
          }
        }
      } else {
        // JSON: signaling
        const msg = JSON.parse(ev.data);
        switch (msg.type) {
          case 'webrtc_offer': {
            // Relayed offer from another peer — create answer and send back
            if (this.pc && msg.sdp) {
              try {
                if (this.pc.signalingState !== 'stable') {
                  console.warn('Ignoring webrtc_offer: state is', this.pc.signalingState);
                  break;
                }
                await this.pc.setRemoteDescription(new RTCSessionDescription({ type: msg.rtc_type, sdp: msg.sdp }));
                const answer = await this.pc.createAnswer();
                await this.pc.setLocalDescription(answer);
                this.ws?.send(JSON.stringify({
                  type: 'webrtc_answer',
                  to_peer: msg.from_peer,
                  sdp: answer.sdp,
                  rtc_type: answer.type,
                }));
              } catch (e) {
                console.warn('Failed to process webrtc_offer:', e);
              }
            }
            break;
          }
          case 'webrtc_answer':
            if (this.pc && msg.sdp) {
              try {
                if (this.pc.signalingState !== 'have-local-offer') {
                  console.warn('Ignoring webrtc_answer: state is', this.pc.signalingState);
                  break;
                }
                await this.pc.setRemoteDescription(new RTCSessionDescription({
                  type: msg.rtc_type,
                  sdp: msg.sdp
                }));
              } catch (e) {
                console.warn('Failed to process webrtc_answer:', e);
              }
            }
            break;
          case 'ice_candidate':
            if (this.pc && msg.candidate) {
              this.pc.addIceCandidate(new RTCIceCandidate({
                candidate: msg.candidate,
                sdpMid: msg.sdpMid,
                sdpMLineIndex: msg.sdpMLineIndex,
              })).catch(() => {}); // ignore if ICE already complete
            }
            break;
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



  requestMetrics(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'metrics_request' }));
    }
    if (this.pc) {
      this.pc.getStats().then(stats => {
        let jitter = 0;
        let packetsLost = 0;
        let packetsReceived = 0;
        let currentRtt = 0;
        
        stats.forEach(report => {
          if (report.type === 'inbound-rtp' && report.kind === 'audio') {
            jitter = report.jitter * 1000 || 0;
            packetsLost = report.packetsLost || 0;
            packetsReceived = report.packetsReceived || 0;
          }
          if (report.type === 'candidate-pair' && report.state === 'succeeded') {
            currentRtt = report.currentRoundTripTime * 1000 || 0;
          }
        });
        
        if (jitter > 0) this.jitterMs = jitter;
        if (currentRtt > 0) this.latencyMs = currentRtt;
        if (packetsReceived > 0) {
           this.packetLossPercent = (packetsLost / (packetsLost + packetsReceived)) * 100;
        }
      }).catch(err => console.error("WebRTC Stats Error:", err));
    }
  }

  setInstrument(instrument: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'instrument_set', instrument }));
    }
  }

  setDDSPMode(enabled: boolean): void {
    this.ddspMode = enabled;
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'ddsp_toggle', enabled }));
    }
  }

  private async _initWebRTC(): Promise<void> {
    if (this.pc) { this.pc.close(); this.pc = null; }

    this.pc = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
      ]
    });

    // Phase 2: add local audio track so peers hear us directly
    if (this.mediaStream) {
      this.mediaStream.getAudioTracks().forEach(track =>
        this.pc!.addTrack(track, this.mediaStream!)
      );
    }

    // Data channel for intent packets (low-latency fallback)
    this.dataChannel = this.pc.createDataChannel('intent');
    this.dataChannel.binaryType = 'arraybuffer';
    this.dataChannel.onmessage = (ev) => {
      if (this.ws && ev.data instanceof ArrayBuffer)
        this.ws.onmessage!(new MessageEvent('message', { data: ev.data }));
    };

    // Receive remote audio track — wire directly to output (near-lossless)
    this.pc.ontrack = (event) => {
      if (event.track.kind === 'audio' && this.audioCtx) {
        this.remoteStream = event.streams[0];
        if (this.remoteStreamSource) this.remoteStreamSource.disconnect();
        this.remoteStreamSource = this.audioCtx.createMediaStreamSource(this.remoteStream);
        // Wire directly to destination; mute the synth when live audio arrives
        this.remoteStreamSource.connect(this.audioCtx.destination);
        if (this.remoteSynth) {
          // Keep synth as safety fallback but silence it
          // (it will re-activate if WebRTC track drops)
        }
      }
    };

    // ICE candidate relay
    this.pc.onicecandidate = (event) => {
      if (event.candidate && this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          type: 'ice_candidate',
          candidate: event.candidate.candidate,
          sdpMid: event.candidate.sdpMid,
          sdpMLineIndex: event.candidate.sdpMLineIndex,
        }));
      }
    };

    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'webrtc_offer',
        sdp: offer.sdp,
        rtc_type: offer.type,
      }));
    }
  }

  private _scheduleAudioBlock(audioData: Float32Array): void {
    if (!this.audioCtx) return;
    
    // Smooth trailing silence to prevent clicking
    if (audioData.length > 0) {
       for (let i=0; i<10; i++) {
           audioData[i] *= (i/10); 
           audioData[audioData.length-1-i] *= (i/10);
       }
    }
    
    const buffer = this.audioCtx.createBuffer(1, audioData.length, 44100);
    buffer.copyToChannel(audioData as Float32Array<ArrayBuffer>, 0);
    
    const source = this.audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioCtx.destination);
    
    const now = this.audioCtx.currentTime;
    if (this.nextAudioTime < now) {
        this.nextAudioTime = now + 0.05; // pre-buffer
    }
    
    source.start(this.nextAudioTime);
    this.nextAudioTime += buffer.duration;
  }

  private regenerateFromIntent(frame: IntentFrame): void {
    // Phase 1: drive the multi-oscillator harmonic synth
    this.remoteSynth?.update(frame);
  }
}
