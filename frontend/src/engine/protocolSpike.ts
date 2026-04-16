/**
 * protocolSpike.ts
 * 
 * Serialization spike for the Dispatched Delta-Capturing Architecture.
 * Proves we can Multiplex two distinct packet types (Pivot vs Delta) 
 * over the same ArrayBuffer WebRTC channel while keeping exact byte sizes
 * and latency constraints in check.
 */

// ─── TYPES ─────────────────────────────────────────────────────────────────

export interface PivotFrame {
  type: 'pivot';
  seq: number;
  timestamp: number;
  f0Hz: number;
  loudnessNorm: number;
  spectralCentroid: number;
  isOnset: boolean;
}

export const N_MELS = 64;

export interface DeltaFrame {
  type: 'delta';
  seq: number;
  ref_seq: number; // The seq of the PivotFrame this refines
  melBands: Float32Array; // 64 dimensions
}

export type NetworkPacket = PivotFrame | DeltaFrame;

// ─── CONSTANTS & OFFSETS ───────────────────────────────────────────────────

// PACKET SIZES
// Base header: 1 byte (type) + 4 bytes (seq) = 5 bytes
export const PIVOT_BYTES = 5 + 4 + 4 + 4 + 4 + 1; // 22 bytes
export const DELTA_BYTES = 5 + 4 + (N_MELS * 4); // 265 bytes

// MAGIC BYTES
const PACKET_TYPE_PIVOT = 0x01;
const PACKET_TYPE_DELTA = 0x02;

// ─── ENCODERS ──────────────────────────────────────────────────────────────

export function encodePacket(packet: NetworkPacket): ArrayBuffer {
  if (packet.type === 'pivot') {
    const buf = new ArrayBuffer(PIVOT_BYTES);
    const view = new DataView(buf);
    let o = 0;
    view.setUint8(o, PACKET_TYPE_PIVOT); o += 1;
    view.setUint32(o, packet.seq, true); o += 4;
    
    view.setFloat32(o, packet.timestamp, true); o += 4;
    view.setFloat32(o, packet.f0Hz, true); o += 4;
    view.setFloat32(o, packet.loudnessNorm, true); o += 4;
    view.setFloat32(o, packet.spectralCentroid, true); o += 4;
    view.setUint8(o, packet.isOnset ? 1 : 0); o += 1;
    return buf;
  } else {
    // Delta Packet
    const buf = new ArrayBuffer(DELTA_BYTES);
    const view = new DataView(buf);
    let o = 0;
    view.setUint8(o, PACKET_TYPE_DELTA); o += 1;
    view.setUint32(o, packet.seq, true); o += 4;
    
    view.setUint32(o, packet.ref_seq, true); o += 4;
    for (let i = 0; i < N_MELS; i++) {
        view.setFloat32(o, packet.melBands[i] || 0, true); o += 4;
    }
    return buf;
  }
}

// ─── DECODERS ──────────────────────────────────────────────────────────────

export function decodePacket(data: ArrayBuffer): NetworkPacket | null {
  if (data.byteLength < 5) return null; // Header too small
  
  const view = new DataView(data);
  let o = 0;
  
  const typeId = view.getUint8(o); o += 1;
  const seq = view.getUint32(o, true); o += 4;
  
  if (typeId === PACKET_TYPE_PIVOT) {
    if (data.byteLength < PIVOT_BYTES) return null;
    const timestamp = view.getFloat32(o, true); o += 4;
    const f0Hz = view.getFloat32(o, true); o += 4;
    const loudnessNorm = view.getFloat32(o, true); o += 4;
    const spectralCentroid = view.getFloat32(o, true); o += 4;
    const isOnset = view.getUint8(o) === 1; o += 1;
    
    return { type: 'pivot', seq, timestamp, f0Hz, loudnessNorm, spectralCentroid, isOnset };
  } else if (typeId === PACKET_TYPE_DELTA) {
    if (data.byteLength < DELTA_BYTES) return null;
    const ref_seq = view.getUint32(o, true); o += 4;
    
    // Instead of looping, we can construct the Float32Array directly from the buffer for speed
    // Ensure we account for the byte offset to prevent unaligned read errors
    const melBands = new Float32Array(data.slice(o, o + (N_MELS * 4)));
    
    return { type: 'delta', seq, ref_seq, melBands };
  }
  
  return null;
}

// ─── SIMULATION TEST ───────────────────────────────────────────────────────

export function runSpike() {
  console.log("=========================================");
  console.log(" MULTIPLEXED PROTOCOL SPIKE (64-dim Mel)");
  console.log("=========================================\n");
  
  // 1. Pivot Packet Test
  const pivot: PivotFrame = {
    type: 'pivot',
    seq: 1001,
    timestamp: 450.5,
    f0Hz: 440.0,
    loudnessNorm: 0.85,
    spectralCentroid: 2500.0,
    isOnset: true
  };
  
  const start = performance.now();
  const pivotBuf = encodePacket(pivot);
  const pivotDecoded = decodePacket(pivotBuf) as PivotFrame;
  const t_pivot = (performance.now() - start) * 1000;
  
  console.log(`[PIVOT] Byte Size: ${pivotBuf.byteLength} bytes`);
  console.log(`[PIVOT] Micro-latency (Encode+Decode): ${t_pivot.toFixed(2)} us`);
  console.assert(pivotDecoded.type === 'pivot', "Failed type match");
  console.assert(pivotDecoded.f0Hz === 440.0, "Failed float match");
  
  // 2. Delta Packet Test
  const dummyMels = new Float32Array(64);
  for(let i=0; i<64; i++) dummyMels[i] = Math.random() * 10 - 5;
  
  const delta: DeltaFrame = {
    type: 'delta',
    seq: 1002,
    ref_seq: 1001,
    melBands: dummyMels
  };
  
  const start2 = performance.now();
  const deltaBuf = encodePacket(delta);
  const deltaDecoded = decodePacket(deltaBuf) as DeltaFrame;
  const t_delta = (performance.now() - start2) * 1000;
  
  console.log(`\n[DELTA] Byte Size: ${deltaBuf.byteLength} bytes`);
  console.log(`[DELTA] Micro-latency (Encode+Decode): ${t_delta.toFixed(2)} us`);
  console.assert(deltaDecoded.type === 'delta', "Failed type match");
  console.assert(deltaDecoded.melBands[63] === dummyMels[63], "Failed float array copy");
  
  // WebRTC Analytics
  console.log("\n--- Structural Analysis ---");
  console.log(`Pivot Rate: 250Hz => ${(pivotBuf.byteLength * 250 * 8 / 1000).toFixed(1)} kbps`);
  console.log(`Delta Rate:  60Hz => ${(deltaBuf.byteLength * 60 * 8 / 1000).toFixed(1)} kbps`);
  console.log(`Total Target Bandwidth: ~${((pivotBuf.byteLength * 250 + deltaBuf.byteLength * 60) * 8 / 1000).toFixed(1)} kbps`);
  console.log(`WebRTC Fragmentation Risk: ${deltaBuf.byteLength > 1200 ? 'HIGH' : 'LOW'}`);
  console.log("=========================================\n");
}

// Execute spike
runSpike();
