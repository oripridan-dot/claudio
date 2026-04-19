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
  // Default: real Opus audio — use native <audio> element to bypass Web Audio 
  // silence bugs and guarantee hardware accelerated playback.
  if (engine.remoteStream) {
     if (!engine.remoteAudioElement) {
         engine.remoteAudioElement = new Audio();
         engine.remoteAudioElement.autoplay = true;
         engine.remoteAudioElement.style.display = 'none';
         document.body.appendChild(engine.remoteAudioElement);
     }
     
     if (engine.remoteAudioElement.srcObject !== engine.remoteStream) {
         engine.remoteAudioElement.srcObject = engine.remoteStream;
         engine.remoteAudioElement.play().catch((e: any) => console.warn("Opus audio play blocked by browser:", e));
     }
  }
}
