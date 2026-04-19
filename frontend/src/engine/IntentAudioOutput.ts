export function scheduleAudioBlock(engine: any, audioData: Float32Array): void {
  // Deprecated: No more DDSP scheduled audio rendering.
  // This function is kept strictly for signature compatibility 
  // if other dummy streams use it, but does not render neural audio.
}

/**
 * v4.1 Audio Routing — Strict Opus Sovereignty
 * 1. Microphone principle: never modify the primary audio path
 * 2. Fiber optic principle: Opus at 128kbps is transparent — use it
 * 3. AI edge principle: DDSP is DEAD.
 *
 * Opus audio ON strictly.
 */
export function updateAudioRouting(engine: any): void {
  // Default: real Opus audio — connect remote stream to speakers
  if (engine.remoteStreamSource && engine.audioCtx) {
     try { engine.remoteStreamSource.disconnect(); } catch(e) {}
     if (engine.masterOut) {
       engine.remoteStreamSource.connect(engine.masterOut);
     } else {
       engine.remoteStreamSource.connect(engine.audioCtx.destination);
     }
  }
}
