import { updateAudioRouting } from './IntentAudioOutput';

export function requestMetrics(engine: any): void {
  if (engine.ws?.readyState === WebSocket.OPEN) {
    engine.ws.send(JSON.stringify({ type: 'metrics_request' }));
  }
  if (engine.pc) {
    engine.pc.getStats().then((stats: any) => {
      let jitter = 0;
      let packetsLost = 0;
      let packetsReceived = 0;
      let currentRtt = 0;
      
      stats.forEach((report: any) => {
        if (report.type === 'inbound-rtp' && report.kind === 'audio') {
          jitter = report.jitter * 1000 || 0;
          packetsLost = report.packetsLost || 0;
          packetsReceived = report.packetsReceived || 0;
        }
        if (report.type === 'candidate-pair' && report.state === 'succeeded') {
          currentRtt = report.currentRoundTripTime * 1000 || 0;
        }
      });
      
      if (jitter > 0) engine.jitterMs = jitter;
      if (currentRtt > 0) engine.latencyMs = currentRtt;
      if (packetsReceived > 0) {
         engine.packetLossPercent = (packetsLost / (packetsLost + packetsReceived)) * 100;
      }

      // Apply smart routing (Auto-fallback if network is thrashing)
      updateAudioRouting(engine);
    }).catch((err: any) => console.error("WebRTC Stats Error:", err));
  }
}
