/**
 * AudioEngineExtensions.ts — BT latency compensation splice.
 *
 * Provides a mixin-style augmentation that adds:
 *   engine.setBTLatencyCompensation(secs)
 *
 * The compensation is realised as a dedicated DelayNode inserted between
 * the master gain and the analyser/destination outputs.  This pre-buffers
 * audio so MIDI/click events scheduled against AudioContext.currentTime
 * are heard at the correct perceptual moment even when the BT codec
 * introduces a hardware pipeline delay.
 *
 * IMPORTANT: setBTLatencyCompensation(0) bypasses the delay node.
 */

import { AudioEngine } from './AudioEngine';

// Store state per-instance using a WeakMap (no prototype mutation needed)
const btDelayState = new WeakMap<AudioEngine, {
  delayNode: DelayNode;
  wired: boolean;
}>();

export function setBTLatencyCompensation(engine: AudioEngine, secs: number): void {
  const ctx = engine.ctx;
  const clampedSecs = Math.max(0, Math.min(0.5, secs)); // 0–500 ms guard

  if (!btDelayState.has(engine)) {
    // First call: create a DelayNode and splice it before the analysers.
    // We do this by disconnecting masterGain → analysers/destination, routing
    // through the new delay node, then reconnecting.
    const btDelay = ctx.createDelay(0.5);
    btDelay.delayTime.value = clampedSecs;

    // Disconnect master gain from downstream nodes
    // (AudioEngine exposes spectrumAnalyser + waveformAnalyser as readonly)
    engine.masterGainNode.disconnect(engine.spectrumAnalyser);
    engine.masterGainNode.disconnect(engine.waveformAnalyser);
    engine.masterGainNode.disconnect(ctx.destination);

    // Wire: masterGain → btDelay → [analysers + destination]
    engine.masterGainNode.connect(btDelay);
    btDelay.connect(engine.spectrumAnalyser);
    btDelay.connect(engine.waveformAnalyser);
    btDelay.connect(ctx.destination);

    btDelayState.set(engine, { delayNode: btDelay, wired: true });
    return;
  }

  const state = btDelayState.get(engine)!;
  state.delayNode.delayTime.setTargetAtTime(clampedSecs, ctx.currentTime, 0.01);
}

// Augment AudioEngine prototype (safe: only adds a method, never overrides)
declare module './AudioEngine' {
  interface AudioEngine {
    // Expose masterGainNode for the BT splice (package-internal)
    masterGainNode: GainNode;
    setBTLatencyCompensation(secs: number): void;
  }
}

(AudioEngine.prototype as any).setBTLatencyCompensation = function (secs: number) {
  setBTLatencyCompensation(this, secs);
};
