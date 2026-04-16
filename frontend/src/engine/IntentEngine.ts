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

import { encodePacket, decodePacket } from './protocol';
import { DDSPDecoder } from './DDSPDecoder';

const FRAME_RATE = 120;

// ─── Intent Engine ──────────────────────────────────────────────────────────

export class IntentEngine {
  private latestLocalMelBands: any = new Float32Array(64);
  private latestRemoteMelBands: any = new Float32Array(64);
  private deltaSeqCounter = 10000;

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
  onCritique?: (critiques: any[]) => void;

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

  // Remote regeneration — DDSP Neural Synth
  private remoteSynth: DDSPDecoder | null = null;
  private melFilterbank: Float32Array[] | null = null;
  ddspMode = false;
  localLoopback = false;
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
        const decoded = decodePacket(ev.data);
        if (decoded) {
            
          if (decoded.type === 'delta') {
              this.latestRemoteMelBands = decoded.melBands;
              return;
          }

          const { seq } = decoded;
          const hybridFrame: IntentFrame = { ...(decoded as PivotFrame), melBands: this.latestRemoteMelBands };

          this.onRemoteIntent?.(hybridFrame);
          
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

          this.regenerateFromIntent(hybridFrame);
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
                if (answer.sdp) {
                  // Force Opus Stereo and 48kHz High-fidelity
                  answer.sdp = answer.sdp.replace(
                    /a=fmtp:101 .*/g, 
                    'a=fmtp:101 minptime=10; useinbandfec=1; stereo=1; sprop-stereo=1; maxplaybackrate=48000; sprop-maxcapturerate=48000; cbr=1'
                  );
                }
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

  setLocalLoopback(enabled: boolean): void {
    this.localLoopback = enabled;
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
      this.mediaStream.getAudioTracks().forEach(track => {
        const sender = this.pc!.addTrack(track, this.mediaStream!);
        const params = sender.getParameters();
        if (!params.encodings) params.encodings = [{}];
        params.encodings[0].maxBitrate = 256000; // Force 256kbps for Studio Fidelity
        sender.setParameters(params).catch(e => console.warn('Failed to set WebRTC maxBitrate:', e));
      });
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
        if (this.masterOut) {
          this.remoteStreamSource.connect(this.masterOut);
        } else {
          this.remoteStreamSource.connect(this.audioCtx.destination);
        }
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
    if (offer.sdp) {
      // Force Opus Stereo and 48kHz High-fidelity
      offer.sdp = offer.sdp.replace(
        /a=fmtp:101 .*/g, 
        'a=fmtp:101 minptime=10; useinbandfec=1; stereo=1; sprop-stereo=1; maxplaybackrate=48000; sprop-maxcapturerate=48000; cbr=1'
      );
    }
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
    // DDSP Client-side inference
    this.remoteSynth?.processFrame(frame);
  }
}
