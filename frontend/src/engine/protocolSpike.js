"use strict";
/**
 * protocolSpike.ts
 *
 * Serialization spike for the Dispatched Delta-Capturing Architecture.
 * Proves we can Multiplex two distinct packet types (Pivot vs Delta)
 * over the same ArrayBuffer WebRTC channel while keeping exact byte sizes
 * and latency constraints in check.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DELTA_BYTES = exports.PIVOT_BYTES = exports.N_MELS = void 0;
exports.encodePacket = encodePacket;
exports.decodePacket = decodePacket;
exports.runSpike = runSpike;
exports.N_MELS = 64;
// ─── CONSTANTS & OFFSETS ───────────────────────────────────────────────────
// PACKET SIZES
// Base header: 1 byte (type) + 4 bytes (seq) = 5 bytes
exports.PIVOT_BYTES = 5 + 4 + 4 + 4 + 4 + 1; // 22 bytes
exports.DELTA_BYTES = 5 + 4 + (exports.N_MELS * 4); // 265 bytes
// MAGIC BYTES
var PACKET_TYPE_PIVOT = 0x01;
var PACKET_TYPE_DELTA = 0x02;
// ─── ENCODERS ──────────────────────────────────────────────────────────────
function encodePacket(packet) {
    if (packet.type === 'pivot') {
        var buf = new ArrayBuffer(exports.PIVOT_BYTES);
        var view = new DataView(buf);
        var o = 0;
        view.setUint8(o, PACKET_TYPE_PIVOT);
        o += 1;
        view.setUint32(o, packet.seq, true);
        o += 4;
        view.setFloat32(o, packet.timestamp, true);
        o += 4;
        view.setFloat32(o, packet.f0Hz, true);
        o += 4;
        view.setFloat32(o, packet.loudnessNorm, true);
        o += 4;
        view.setFloat32(o, packet.spectralCentroid, true);
        o += 4;
        view.setUint8(o, packet.isOnset ? 1 : 0);
        o += 1;
        return buf;
    }
    else {
        // Delta Packet
        var buf = new ArrayBuffer(exports.DELTA_BYTES);
        var view = new DataView(buf);
        var o = 0;
        view.setUint8(o, PACKET_TYPE_DELTA);
        o += 1;
        view.setUint32(o, packet.seq, true);
        o += 4;
        view.setUint32(o, packet.ref_seq, true);
        o += 4;
        for (var i = 0; i < exports.N_MELS; i++) {
            view.setFloat32(o, packet.melBands[i] || 0, true);
            o += 4;
        }
        return buf;
    }
}
// ─── DECODERS ──────────────────────────────────────────────────────────────
function decodePacket(data) {
    if (data.byteLength < 5)
        return null; // Header too small
    var view = new DataView(data);
    var o = 0;
    var typeId = view.getUint8(o);
    o += 1;
    var seq = view.getUint32(o, true);
    o += 4;
    if (typeId === PACKET_TYPE_PIVOT) {
        if (data.byteLength < exports.PIVOT_BYTES)
            return null;
        var timestamp = view.getFloat32(o, true);
        o += 4;
        var f0Hz = view.getFloat32(o, true);
        o += 4;
        var loudnessNorm = view.getFloat32(o, true);
        o += 4;
        var spectralCentroid = view.getFloat32(o, true);
        o += 4;
        var isOnset = view.getUint8(o) === 1;
        o += 1;
        return { type: 'pivot', seq: seq, timestamp: timestamp, f0Hz: f0Hz, loudnessNorm: loudnessNorm, spectralCentroid: spectralCentroid, isOnset: isOnset };
    }
    else if (typeId === PACKET_TYPE_DELTA) {
        if (data.byteLength < exports.DELTA_BYTES)
            return null;
        var ref_seq = view.getUint32(o, true);
        o += 4;
        // Instead of looping, we can construct the Float32Array directly from the buffer for speed
        // Ensure we account for the byte offset to prevent unaligned read errors
        var melBands = new Float32Array(data.slice(o, o + (exports.N_MELS * 4)));
        return { type: 'delta', seq: seq, ref_seq: ref_seq, melBands: melBands };
    }
    return null;
}
// ─── SIMULATION TEST ───────────────────────────────────────────────────────
function runSpike() {
    console.log("=========================================");
    console.log(" MULTIPLEXED PROTOCOL SPIKE (64-dim Mel)");
    console.log("=========================================\n");
    // 1. Pivot Packet Test
    var pivot = {
        type: 'pivot',
        seq: 1001,
        timestamp: 450.5,
        f0Hz: 440.0,
        loudnessNorm: 0.85,
        spectralCentroid: 2500.0,
        isOnset: true
    };
    var start = performance.now();
    var pivotBuf = encodePacket(pivot);
    var pivotDecoded = decodePacket(pivotBuf);
    var t_pivot = (performance.now() - start) * 1000;
    console.log("[PIVOT] Byte Size: ".concat(pivotBuf.byteLength, " bytes"));
    console.log("[PIVOT] Micro-latency (Encode+Decode): ".concat(t_pivot.toFixed(2), " us"));
    console.assert(pivotDecoded.type === 'pivot', "Failed type match");
    console.assert(pivotDecoded.f0Hz === 440.0, "Failed float match");
    // 2. Delta Packet Test
    var dummyMels = new Float32Array(64);
    for (var i = 0; i < 64; i++)
        dummyMels[i] = Math.random() * 10 - 5;
    var delta = {
        type: 'delta',
        seq: 1002,
        ref_seq: 1001,
        melBands: dummyMels
    };
    var start2 = performance.now();
    var deltaBuf = encodePacket(delta);
    var deltaDecoded = decodePacket(deltaBuf);
    var t_delta = (performance.now() - start2) * 1000;
    console.log("\n[DELTA] Byte Size: ".concat(deltaBuf.byteLength, " bytes"));
    console.log("[DELTA] Micro-latency (Encode+Decode): ".concat(t_delta.toFixed(2), " us"));
    console.assert(deltaDecoded.type === 'delta', "Failed type match");
    console.assert(deltaDecoded.melBands[63] === dummyMels[63], "Failed float array copy");
    // WebRTC Analytics
    console.log("\n--- Structural Analysis ---");
    console.log("Pivot Rate: 250Hz => ".concat((pivotBuf.byteLength * 250 * 8 / 1000).toFixed(1), " kbps"));
    console.log("Delta Rate:  60Hz => ".concat((deltaBuf.byteLength * 60 * 8 / 1000).toFixed(1), " kbps"));
    console.log("Total Target Bandwidth: ~".concat(((pivotBuf.byteLength * 250 + deltaBuf.byteLength * 60) * 8 / 1000).toFixed(1), " kbps"));
    console.log("WebRTC Fragmentation Risk: ".concat(deltaBuf.byteLength > 1200 ? 'HIGH' : 'LOW'));
    console.log("=========================================\n");
}
// Execute if run natively (e.g. via tsx)
if (typeof require !== 'undefined' && require.main === module) {
    runSpike();
}
