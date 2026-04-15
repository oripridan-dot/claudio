import { IntentFrame } from './types';
import { N_MFCC } from './dsp';

export const PACKET_BYTES = 86;

export function encodeIntentPacket(frame: IntentFrame, seq: number): ArrayBuffer {
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

export function decodeIntentPacket(data: ArrayBuffer): { seq: number; frame: IntentFrame } | null {
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
