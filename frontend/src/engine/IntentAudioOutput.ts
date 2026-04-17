export function scheduleAudioBlock(engine: any, audioData: Float32Array): void {
  if (!engine.audioCtx) return;
  // Smooth trailing silence to prevent clicking
  if (audioData.length > 0) {
     for (let i = 0; i < 10; i++) {
         audioData[i] *= (i / 10); 
         audioData[audioData.length - 1 - i] *= (i / 10);
     }
  }
  
  const buffer = engine.audioCtx.createBuffer(1, audioData.length, 44100);
  buffer.copyToChannel(audioData as Float32Array<ArrayBuffer>, 0);
  
  const source = engine.audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(engine.audioCtx.destination);
  
  const now = engine.audioCtx.currentTime;
  if (engine.nextAudioTime < now) {
      engine.nextAudioTime = now + 0.05; // pre-buffer
  }
  
  source.start(engine.nextAudioTime);
  engine.nextAudioTime += buffer.duration;
}

/**
 * v4.0 Audio Routing — Three Laws of Claudio Audio:
 * 1. Microphone principle: never modify the primary audio path
 * 2. Fiber optic principle: Opus at 128kbps is transparent — use it
 * 3. AI edge principle: DDSP is emergency fallback ONLY
 *
 * Default: Opus audio ON, DDSP synths OFF
 * Emergency: DDSP activates only when network is catastrophically degraded
 */
export function updateAudioRouting(engine: any): void {
  const isNetworkCatastrophic = engine.packetLossPercent > 25 && engine.jitterMs > 200;
  const hasRealAudio = !!engine.remoteStreamSource;
  const needsDDSPFallback = engine.ddspMode || (isNetworkCatastrophic && !hasRealAudio);

  if (needsDDSPFallback && !hasRealAudio) {
    // Emergency: no real audio available, activate DDSP neural synth
    engine.remoteSynths.forEach((s: any) => s.setMuted(false));
  } else {
    // Default: real Opus audio — connect remote stream to speakers
    if (engine.remoteStreamSource && engine.audioCtx) {
       try { engine.remoteStreamSource.disconnect(); } catch(e) {}
       if (engine.masterOut) {
         engine.remoteStreamSource.connect(engine.masterOut);
       } else {
         engine.remoteStreamSource.connect(engine.audioCtx.destination);
       }
    }
    // Mute all DDSP synths — real audio is playing
    engine.remoteSynths.forEach((s: any) => s.setMuted(true));
  }
}
