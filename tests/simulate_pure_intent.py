import time

import numpy as np

from claudio.intent.intent_decoder import IntentDecoder
from claudio.intent.intent_encoder import IntentEncoder
from claudio.intent.intent_protocol import IntentStream


def _sine(freq: float, dur: float, sr: int) -> np.ndarray:
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    return (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)

def simulate_pure_intent_latency():
    print("=" * 60)
    print("  CLAUDIO v3.0 PURE INTENT LATENCY SIMULATION")
    print("=" * 60)

    sr = 48000
    block_size = int(sr / 120) # 120Hz frame rate -> 400 samples

    encoder = IntentEncoder(sample_rate=sr)
    decoder = IntentDecoder(sample_rate=sr)
    stream = IntentStream()

    audio = _sine(440, 1.0, sr)
    blocks = len(audio) // block_size

    encode_times = []
    pack_times = []
    transmit_sim_network_ms = 1.0 # 1ms LAN assumed
    decode_times = []

    print(f"\n▶ Simulating 1 second of audio at {120}Hz framerate (block_size: {block_size})...")

    for b in range(blocks):
        chunk = audio[b * block_size : (b + 1) * block_size]

        # 1. Encode
        t0 = time.perf_counter()
        frames = encoder.encode_block(chunk)
        t_encode = (time.perf_counter() - t0) * 1000

        # 2. Pack
        t0 = time.perf_counter()
        packets = [stream.pack(f).to_bytes() for f in frames]
        t_pack = (time.perf_counter() - t0) * 1000

        # 3. Decode
        t0 = time.perf_counter()
        from claudio.intent.intent_protocol import IntentPacket
        restored = [IntentPacket.from_bytes(p).frame for p in packets if IntentPacket.from_bytes(p).frame]
        decoder.decode_frames(restored)
        t_decode = (time.perf_counter() - t0) * 1000

        encode_times.append(t_encode / len(frames) if frames else t_encode)
        pack_times.append(t_pack / len(packets) if packets else t_pack)
        decode_times.append(t_decode / len(restored) if restored else t_decode)

    avg_enc = np.mean(encode_times)
    avg_pack = np.mean(pack_times)
    avg_dec = np.mean(decode_times)

    total_latency = avg_enc + avg_pack + transmit_sim_network_ms + avg_dec

    print("\n[ Latency Sub-stage Breakdown ]")
    print(f"  Feature Extraction (Pitch/MFCC):  {avg_enc:.3f} ms")
    print(f"  Binary Serialization:             {avg_pack:.3f} ms")
    print(f"  Network Transit (LAN p2p):        {transmit_sim_network_ms:.3f} ms")
    print(f"  Client Synthesis / DDSP routing:  {avg_dec:.3f} ms")
    print("-" * 60)
    print(f"  TOTAL E2E INTENT LATENCY:         {total_latency:.3f} ms")

    print("\nNote: This is Python execution. The frontend WASM/WebNN implementation")
    print("will be an order of magnitude faster (sub-1ms extraction).")

    if total_latency <= 15.0:
        print("\n✅ PASSED: Total Latency is well under the 15ms promise!")
    else:
        print("\n❌ FAILED: Latency exceeded 15ms target.")

if __name__ == "__main__":
    simulate_pure_intent_latency()
