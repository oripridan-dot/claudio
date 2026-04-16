import { NetworkPacket, PivotFrame, DeltaFrame } from './types';

// PACKET SIZES
export const PIVOT_BYTES = 5 + 4 + 4 + 4 + 4 + 1; // 22 bytes
export const DELTA_BYTES = 5 + 4 + (64 * 4); // 265 bytes

// MAGIC BYTES
export const PACKET_TYPE_PIVOT = 0x01;
export const PACKET_TYPE_DELTA = 0x02;

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
    for (let i = 0; i < 64; i++) {
        view.setFloat32(o, packet.melBands?.[i] || 0, true); o += 4;
    }
    return buf;
  }
}

export function decodePacket(data: ArrayBuffer): NetworkPacket | null {
  if (data.byteLength < 5) return null;
  
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
    
    return { type: 'pivot', seq, timestamp, f0Hz, loudnessNorm, spectralCentroid, isOnset } as PivotFrame;
  } else if (typeId === PACKET_TYPE_DELTA) {
    if (data.byteLength < DELTA_BYTES) return null;
    const ref_seq = view.getUint32(o, true); o += 4;
    
    // Construct Float32Array directly off the ArrayBuffer buffer cache
    const melBands = new Float32Array(data.slice(o, o + (64 * 4)));
    
    return { type: 'delta', seq, ref_seq, melBands } as DeltaFrame;
  }
  
  return null;
}
