import type { IntentEngine } from './IntentEngine';
import type { IntentFrame, PivotFrame } from './types';
import { decodePacket } from './protocol';

export function initIntentWebSocket(engine: any) {

    if (!engine.jwtToken) return;
    const wsUrl = `${engine.serverUrl.replace('http', 'ws')}/ws/collab/${engine.roomId}?name=${encodeURIComponent(engine.displayName)}&token=${engine.jwtToken}&instrument=${encodeURIComponent(engine.instrumentProfile)}&environment=${encodeURIComponent(engine.environmentProfile)}`;
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

        // Binary: remote intent packet with 8-byte UDP-style header
        const dataBuf = ev.data as ArrayBuffer;
        if (dataBuf.byteLength < 8) return;
        
        const pidView = new Uint8Array(dataBuf, 0, 8);
        const peerId = new TextDecoder().decode(pidView).replace(/\0/g, '');
        const intentData = dataBuf.slice(8);

        const decoded = decodePacket(intentData);
        if (decoded) {
            
          if (decoded.type === 'delta') {
              if (!engine.remoteMelBandsCaches) engine.remoteMelBandsCaches = new Map();
              engine.remoteMelBandsCaches.set(peerId, decoded.melBands);
              return;
          }

          const { seq } = decoded;
          const cachedBands = engine.remoteMelBandsCaches?.get(peerId) || new Float32Array(64);
          const hybridFrame: IntentFrame = { ...(decoded as PivotFrame), melBands: cachedBands, peerId };

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
                  // Polite peer pattern: if we have a pending offer, the peer with
                  // the lower ID rolls back to accept the incoming offer.
                  const isPolite = (engine.peerId || '') < (msg.from_peer || '');
                  if (isPolite) {
                    await engine.pc.setLocalDescription({ type: 'rollback' });
                  } else {
                    console.warn('Impolite peer: ignoring conflicting offer');
                    break;
                  }
                }
                await engine.pc.setRemoteDescription(new RTCSessionDescription({ type: msg.rtc_type, sdp: msg.sdp }));
                const answer = await engine.pc.createAnswer();
                if (answer.sdp) {
                  // v4.0 SDP munging: High-fidelity Opus for music (answer)
                  answer.sdp = answer.sdp.replace(
                    /a=fmtp:(\d+) .*/g, 
                    'a=fmtp:$1 minptime=10; useinbandfec=1; stereo=1; sprop-stereo=1; maxaveragebitrate=128000; maxplaybackrate=48000; sprop-maxcapturerate=48000'
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
            engine.peers = msg.peers || [];
            engine.onPeersUpdated?.(engine.peers);
            // Re-initiate WebRTC when a new peer joins so the existing peer
            // sends a fresh offer (solves the glare deadlock where the initial
            // offer was sent to an empty room).
            engine._initWebRTC();
            break;
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

    // v4.0: Audio track is the PRIMARY channel — real Opus audio
    if (engine.mediaStream) {
      engine.mediaStream.getAudioTracks().forEach(track => {
        const sender = engine.pc!.addTrack(track, engine.mediaStream!);
        const params = sender.getParameters();
        if (!params.encodings) params.encodings = [{}];
        params.encodings[0].maxBitrate = 128000; // 128kbps — perceptually transparent Opus
        params.encodings[0].networkPriority = 'high'; // v4.0: Audio is king, not the fallback
        sender.setParameters(params).catch(e => console.warn('Failed to set WebRTC audio params:', e));
      });
    }

    // v4.0: Intent data channel — secondary intelligence path
    engine.dataChannel = engine.pc.createDataChannel('intent', {
      ordered: false,
      maxRetransmits: 0,
      priority: 'low' // v4.0: Intent is the observer, not the primary
    });
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
      // v4.0 SDP munging: High-fidelity Opus for music
      offer.sdp = offer.sdp.replace(
        /a=fmtp:(\d+) .*/g, 
        'a=fmtp:$1 minptime=10; useinbandfec=1; stereo=1; sprop-stereo=1; maxaveragebitrate=128000; maxplaybackrate=48000; sprop-maxcapturerate=48000'
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