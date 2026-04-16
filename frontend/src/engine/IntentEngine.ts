/**
 * IntentEngine.ts — Client-side Intent Capture and WebSocket Streaming
 *
 * Phase 1: Multi-oscillator harmonic synth + MFCC timbre packets
 * Phase 2: WebRTC sendrecv audio + ICE relay → near-lossless p2p audio
 */

import {
  IntentFrame,
  PeerInfo,
  CollabMetrics,
  IntentCallback,
  PeerCallback,
  MetricsCallback,
  PivotFrame,
  DeltaFrame
} from './types';

export type {
  IntentFrame,
  PeerInfo,
  CollabMetrics,
  IntentCallback,
  PeerCallback,
  MetricsCallback
};

import {
  initWasmDSP,
  autocorrelationF0,
  computeRMS,
  computeSpectralCentroid,
  buildMelFilterbank,
  computeMelBands
} from './dsp';

import { initIntentWebSocket, initIntentWebRTC } from './IntentWebRTC';
import { encodePacket, decodePacket } from './protocol';
import { DDSPDecoder } from './DDSPDecoder';
import { RTCalibrationEngine } from './RTCalibrationEngine';

const FRAME_RATE = 120;

// ─── Intent Engine ──────────────────────────────────────────────────────────

export class IntentEngine {
  private rcEngine: RTCalibrationEngine | null = null;
  private latestLocalMelBands: any = new Float32Array(64);
  latestRemoteMelBands: any = new Float32Array(64);
  private deltaSeqCounter = 10000;

  private audioCtx: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private analyser: AnalyserNode | null = null;
  ws: WebSocket | null = null;

  private isCapturing = false;
  private captureInterval: number | null = null;
  private seq = 0;

  // Callbacks
  onLocalIntent: IntentCallback | null = null;
  onRemoteIntent: IntentCallback | null = null;
  onPeersUpdated: PeerCallback | null = null;
  onMetrics: MetricsCallback | null = null;
  onConnectionChange: ((connected: boolean) => void) | null = null;
  onCritique?: (critiques: any[]) => void;

  // WebRTC state
  pc: RTCPeerConnection | null = null;
  dataChannel: RTCDataChannel | null = null;
  remoteStream: MediaStream | null = null;
  remoteStreamSource: MediaStreamAudioSourceNode | null = null;

  // State
  peerId = '';
  roomId = '';
  peers: PeerInfo[] = [];
  connected = false;

  // Remote regeneration — DDSP Neural Synth
  private remoteSynth: DDSPDecoder | null = null;
  private melFilterbank: Float32Array[] | null = null;
  ddspMode = false;
  localLoopback = false;
  nextAudioTime = 0;

  // Onset detection state
  private prevRMS = 0;

  // Reconnection state
  reconnectAttempts = 0;
  maxReconnectAttempts = 5;
  reconnectTimer: number | null = null;
  serverUrl = '';
  displayName = '';
  jwtToken: string | null = null;

  // Audio output routing
  masterOut: GainNode | null = null;

  // Shadow validation pipeline (Learning Kit)
  private shadowWorker: Worker | null = null;

  getAudioContext(): AudioContext | null { return this.audioCtx; }
  getInputAnalyser(): AnalyserNode | null { return this.analyser; }
  
  redirectOutput(target: AudioNode) {
    if (this.masterOut && this.audioCtx) {
      this.masterOut.disconnect();
      this.masterOut.connect(target);
    }
  }

  // Latency & Network Telemetry logging
  lastPingTs = 0;
  latencyMs = 0;
  
  jitterMs = 0.0;
  packetLossPercent = 0.0;
  remoteSeq: number | null = null;
  packetsReceived = 0;
  packetsLost = 0;
  packetsWindowReceived = 0;
  packetsWindowLost = 0;
  lastNetworkUpdateTs = 0;

  async startCapture(): Promise<void> {
    this.audioCtx = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });
    
    this.masterOut = this.audioCtx.createGain();
    this.masterOut.connect(this.audioCtx.destination);

    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { 
        echoCancellation: false, 
        noiseSuppression: false, 
        autoGainControl: false,
        channelCount: 2,
        sampleRate: 48000
      },
    });
    
    if (this.audioCtx.state === 'suspended') {
      await this.audioCtx.resume();
    }

    const source = this.audioCtx.createMediaStreamSource(this.mediaStream);
    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 2048;
    source.connect(this.analyser);

    // Build mel filterbank once (2048-point FFT, 48000 Hz) — stores internally in dsp.ts module
    buildMelFilterbank(2048, 48000);
    this.melFilterbank = []; // Mark as initialized — actual state lives in dsp.ts module

    // Ensure WASM core is ready for sub-millisecond intent extraction
    await initWasmDSP();

    // High-fidelity neural DDSP for local loopback and remote regeneration
    this.remoteSynth = new DDSPDecoder(this.audioCtx);
    await this.remoteSynth.loadModel();

    // Real-Time Auto-Calibration Loop (Ghost Observer)
    this.rcEngine = new RTCalibrationEngine(this.audioCtx);
    this.rcEngine.connectInputSource(source);
    this.rcEngine.connectOutputTap(this.remoteSynth.getOutputNode());
    
    this.rcEngine.onAutoCalibrate = (metrics) => {
        if (!this.remoteSynth) return;
        
        // Scenario A: The Bleeding Gate (Unvoiced DC Rumble Tracking)
        if (metrics.inputMagnitudeDb < -55 && metrics.outputMagnitudeDb > -35) {
            this.remoteSynth.setGateOverride(0.01);
        } else {
            this.remoteSynth.setGateOverride(1.0);
        }

        // Scenario B: The Muffled Synth (Brightness Shift)
        if (metrics.inputCentroidHz > 3000 && metrics.outputCentroidHz < 1500) {
            this.remoteSynth.setNoiseMultiplier(1.5);
        } else {
            this.remoteSynth.setNoiseMultiplier(1.0);
        }
    };

    // [Learning Kit] Instantiate Shadow Worker if in teaching environment
    if (import.meta.env.VITE_APP_ENV === 'teaching') {
      this.shadowWorker = new Worker(new URL('./learning/ValidationWorker.ts', import.meta.url), { type: 'module' });
      this.shadowWorker.postMessage({ type: 'init' });
      
      this.shadowWorker.onmessage = (e) => {
        if (e.data.type === 'critiques') {
          if (this.onCritique) {
             this.onCritique(e.data.critiques);
          }
          e.data.critiques.forEach((c: any) => {
            // Log to console for backup
            console.warn(`[BRUTAL HONESTY] ${c.message} (Delta: ${c.delta.toFixed(2)})`);
          });
        }
      };
    }

    this.isCapturing = true;

    this.captureInterval = window.setInterval(() => {
      if (!this.analyser || !this.audioCtx) return;

      this.rcEngine?.updateMetrics();

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
      
      const rmsJump = rms - this.prevRMS;
      const isOnset = rmsJump > 0.05 && rms > 0.02;
      this.prevRMS = rms;

      this.seq++;

      const pivot: PivotFrame = {
        type: 'pivot',
        seq: this.seq,
        timestamp: performance.now(),
        f0Hz: f0, confidence, loudnessDb, loudnessNorm,
        spectralCentroid: centroid, isOnset,
        onsetStrength: Math.max(0, rmsJump),
        rmsEnergy: rms,
      };

      const isDeltaCycle = (this.seq % 4 === 0) || isOnset;
      if (isDeltaCycle) {
        Promise.resolve().then(() => {
            const melBands = computeMelBands(magSpectrum);
            this.latestLocalMelBands = melBands;
            this.deltaSeqCounter++;
            const delta: DeltaFrame = {
                type: 'delta',
                seq: this.deltaSeqCounter,
                ref_seq: pivot.seq,
                timestamp: pivot.timestamp,
                melBands
            };
            if (this.dataChannel?.readyState === 'open') {
                this.dataChannel.send(encodePacket(delta));
            } else if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.send(encodePacket(delta));
            }
        });
      }

      const hybridFrame: IntentFrame = { ...pivot, melBands: this.latestLocalMelBands };

      this.onLocalIntent?.(hybridFrame);
      
      if (this.localLoopback) {
          this.regenerateFromIntent(hybridFrame);
      }

      // [Learning Kit] Post frame off main-thread to be ruthlessly evaluated
      if (this.shadowWorker) {
        this.shadowWorker.postMessage({ type: 'frame', frame: hybridFrame });
      }

      if (this.dataChannel?.readyState === 'open') {
        this.dataChannel.send(encodePacket(pivot));
      } else if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(encodePacket(pivot));
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
    if (this.shadowWorker) {
      this.shadowWorker.terminate();
      this.shadowWorker = null;
    }
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

  _connectWebSocket(): void {
    initIntentWebSocket(this);
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

        // Apply smart routing (Auto-fallback if network is thrashing)
        this._updateAudioRouting();
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
    this._updateAudioRouting();
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'ddsp_toggle', enabled }));
    }
  }

  private _updateAudioRouting(): void {
    const isNetworkPoor = this.packetLossPercent > 15 || this.jitterMs > 100;
    const forceDDSP = this.ddspMode || isNetworkPoor || !this.remoteStreamSource;

    if (forceDDSP) {
      // Mute Raw WebRTC Audio
      if (this.remoteStreamSource) {
         try { this.remoteStreamSource.disconnect(); } catch(e) {}
      }
      // Unmute DDSP Synth so Neural Regeneration can seamlessly take over
      this.remoteSynth?.setMuted(false);
    } else {
      // Unmute Raw WebRTC Audio
      if (this.remoteStreamSource && this.audioCtx) {
         try { this.remoteStreamSource.disconnect(); } catch(e) {}
         if (this.masterOut) {
           this.remoteStreamSource.connect(this.masterOut);
         } else {
           this.remoteStreamSource.connect(this.audioCtx.destination);
         }
      }
      // Mute DDSP Synth to prevent horrific double-audio echo
      this.remoteSynth?.setMuted(true);
    }
  }

  setLocalLoopback(enabled: boolean): void {
    this.localLoopback = enabled;
  }

  async _initWebRTC(): Promise<void> {
    await initIntentWebRTC(this);
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
    // DDSP Client-side inference
    this.remoteSynth?.processFrame(frame);
  }
}
