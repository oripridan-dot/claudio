import type { IntentEngine } from './IntentEngine';
import type { IntentFrame, PivotFrame } from './types';
import { decodePacket } from './protocol';

export function initIntentWebSocket(engine: any) {

    if (!engine.jwtToken) return;
    const wsUrl = `${engine.serverUrl.replace('http', 'ws')}/ws/collab/${engine.roomId}?name=${encodeURIComponent(engine.displayName)}&token=${engine.jwtToken}`;
    engine.ws = new WebSocket(wsUrl);
    engine.ws.binaryType = 'arraybuffer';

    engine.ws.onopen = () => {
      engine.connected = true;
      engine.reconnectAttempts = 0;
      engine.onConnectionChange?.(true);

      // Start ping interval for latency measurement
      engine._startPing();
      engine._initWebRTC();
    };

    engine.ws.onclose = () => {
      engine.connected = false;
      engine.onConnectionChange?.(false);
      engine._attemptReconnect();
    };

    engine.ws.onmessage = async (ev) => {

      if (ev.data instanceof ArrayBuffer) {
        const view = new DataView(ev.data);
        if (ev.data.byteLength > 4) {
          const magic = view.getUint32(0, false); // Big endian read of "DDSP"
          if (magic === 0x44445350) { // b"DDSP"
            const audioData = new Float32Array(ev.data.slice(12)); // skip 4 magic + 8 peer_id
            engine._scheduleAudioBlock(audioData);
            return;
          }
        }

        // Binary: remote intent packet
        const decoded = decodePacket(ev.data);
        if (decoded) {
            
          if (decoded.type === 'delta') {
              engine.latestRemoteMelBands = decoded.melBands;
              return;
          }

          const { seq } = decoded;
          const hybridFrame: IntentFrame = { ...(decoded as PivotFrame), melBands: engine.latestRemoteMelBands };

          engine.onRemoteIntent?.(hybridFrame);
          
          // Network Telemetry calculate: Packet Loss
          if (engine.remoteSeq !== null) {
              const expected = engine.remoteSeq + 1;
              if (seq > expected) {
                  const lost = seq - expected;
                  engine.packetsLost += lost;
                  engine.packetsWindowLost += lost;
              }
          }
          engine.remoteSeq = seq;
          engine.packetsReceived++;
          engine.packetsWindowReceived++;

          // Network Telemetry calculate: Jitter
          const now = performance.now();
          if (engine.lastNetworkUpdateTs > 0) {
              const deltaR = now - engine.lastNetworkUpdateTs;
              const deltaS = 8.33; // Nominal at 120Hz
              const diff = Math.abs(deltaR - deltaS);
              // Moving average using RTCP formula J = J + (|D| - J) / 16
              engine.jitterMs += (diff - engine.jitterMs) / 16;
          }
          engine.lastNetworkUpdateTs = now;
          
          // Periodically update packet loss percent (every 1 second approx 120 pkts)
          if (engine.packetsWindowReceived > 120) {
              const totalWindow = engine.packetsWindowReceived + engine.packetsWindowLost;
              engine.packetLossPercent = totalWindow > 0 ? (engine.packetsWindowLost / totalWindow) * 100 : 0;
              engine.packetsWindowReceived = 0;
              engine.packetsWindowLost = 0;
          }

          engine.regenerateFromIntent(hybridFrame);
        }
      } else {
        // JSON: signaling
        const msg = JSON.parse(ev.data);
        switch (msg.type) {
          case 'webrtc_offer': {
            // Relayed offer from another peer — create answer and send back
            if (engine.pc && msg.sdp) {
              try {
                if (engine.pc.signalingState !== 'stable') {
                  console.warn('Ignoring webrtc_offer: state is', engine.pc.signalingState);
                  break;
                }
                await engine.pc.setRemoteDescription(new RTCSessionDescription({ type: msg.rtc_type, sdp: msg.sdp }));
                const answer = await engine.pc.createAnswer();
                if (answer.sdp) {
                  // Force Opus Stereo and 48kHz High-fidelity
                  answer.sdp = answer.sdp.replace(
                    /a=fmtp:101 .*/g, 
                    'a=fmtp:101 minptime=10; useinbandfec=1; stereo=1; sprop-stereo=1; maxplaybackrate=48000; sprop-maxcapturerate=48000; cbr=1'
                  );
                }
                await engine.pc.setLocalDescription(answer);
                engine.ws?.send(JSON.stringify({
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
            if (engine.pc && msg.sdp) {
              try {
                if (engine.pc.signalingState !== 'have-local-offer') {
                  console.warn('Ignoring webrtc_answer: state is', engine.pc.signalingState);
                  break;
                }
                await engine.pc.setRemoteDescription(new RTCSessionDescription({
                  type: msg.rtc_type,
                  sdp: msg.sdp
                }));
              } catch (e) {
                console.warn('Failed to process webrtc_answer:', e);
              }
            }
            break;
          case 'ice_candidate':
            if (engine.pc && msg.candidate) {
              engine.pc.addIceCandidate(new RTCIceCandidate({
                candidate: msg.candidate,
                sdpMid: msg.sdpMid,
                sdpMLineIndex: msg.sdpMLineIndex,
              })).catch(() => {}); // ignore if ICE already complete
            }
            break;
          case 'welcome':
            engine.peerId = msg.peer_id;
            engine.peers = msg.peers || [];
            engine.onPeersUpdated?.(engine.peers);
            break;
          case 'peer_joined':
          case 'peer_updated':
          case 'peer_left':
            engine.peers = msg.peers || [];
            engine.onPeersUpdated?.(engine.peers);
            break;
          case 'metrics':
            engine.onMetrics?.(msg);
            break;
          case 'pong': {
            if (engine.lastPingTs > 0) {
              engine.latencyMs = performance.now() - engine.lastPingTs;
            }
            break;
          }
        }

      }
    };
  }
export async function initIntentWebRTC(engine: any) {

    if (engine.pc) { engine.pc.close(); engine.pc = null; }

    engine.pc = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
      ]
    });

    // Phase 2: add local audio track so peers hear us directly
    if (engine.mediaStream) {
      engine.mediaStream.getAudioTracks().forEach(track => {
        const sender = engine.pc!.addTrack(track, engine.mediaStream!);
        const params = sender.getParameters();
        if (!params.encodings) params.encodings = [{}];
        params.encodings[0].maxBitrate = 256000; // Force 256kbps for Studio Fidelity
        sender.setParameters(params).catch(e => console.warn('Failed to set WebRTC maxBitrate:', e));
      });
    }

    // Data channel for intent packets (low-latency fallback)
    engine.dataChannel = engine.pc.createDataChannel('intent');
    engine.dataChannel.binaryType = 'arraybuffer';
    engine.dataChannel.onmessage = (ev) => {
      if (engine.ws && ev.data instanceof ArrayBuffer)
        engine.ws.onmessage!(new MessageEvent('message', { data: ev.data }));
    };

    // Receive remote audio track — wire directly to output (near-lossless)
    engine.pc.ontrack = (event) => {
      if (event.track.kind === 'audio' && engine.audioCtx) {
        engine.remoteStream = event.streams[0];
        if (engine.remoteStreamSource) engine.remoteStreamSource.disconnect();
        engine.remoteStreamSource = engine.audioCtx.createMediaStreamSource(engine.remoteStream);
        
        // Smart Routing trigger: dynamically decide whether to map this directly to speakers 
        // or keep relying on DDSP fallback depending on ddspMode and line quality.
        engine._updateAudioRouting();
      }
    };

    // ICE candidate relay
    engine.pc.onicecandidate = (event) => {
      if (event.candidate && engine.ws?.readyState === WebSocket.OPEN) {
        engine.ws.send(JSON.stringify({
          type: 'ice_candidate',
          candidate: event.candidate.candidate,
          sdpMid: event.candidate.sdpMid,
          sdpMLineIndex: event.candidate.sdpMLineIndex,
        }));
      }
    };

    const offer = await engine.pc.createOffer();
    if (offer.sdp) {
      // Force Opus Stereo and 48kHz High-fidelity
      offer.sdp = offer.sdp.replace(
        /a=fmtp:101 .*/g, 
        'a=fmtp:101 minptime=10; useinbandfec=1; stereo=1; sprop-stereo=1; maxplaybackrate=48000; sprop-maxcapturerate=48000; cbr=1'
      );
    }
    await engine.pc.setLocalDescription(offer);

    if (engine.ws?.readyState === WebSocket.OPEN) {
      engine.ws.send(JSON.stringify({
        type: 'webrtc_offer',
        sdp: offer.sdp,
        rtc_type: offer.type,
      }));
    }
  }