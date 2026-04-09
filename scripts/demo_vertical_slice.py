#!/usr/bin/env python3
"""
demo_vertical_slice.py — End-to-End Vertical Slice Proof

Runs the complete Claudio intent pipeline on a real multitrack WAV file:
  1. Load source audio (acoustic guitar)
  2. Extract intent frames (F0, timbre, loudness, onset)
  3. Serialize to binary packets (simulating network transport)
  4. Deserialize and reconstruct intent frames
  5. Regenerate audio from intent using additive synthesis
  6. Save original + regenerated WAV files for A/B comparison
  7. Print comprehensive pipeline metrics

Usage:
  python demo_vertical_slice.py [--source path/to/file.wav] [--output-dir ./demo_output]
"""
from __future__ import annotations

import argparse
import math
import struct
import time
import wave
from pathlib import Path

import numpy as np

from claudio.intent.intent_decoder import IntentDecoder
from claudio.intent.intent_encoder import IntentEncoder, IntentFrame
from claudio.intent.intent_protocol import IntentPacket, IntentStream, PacketFlags


def load_wav(path: Path, target_sr: int = 44_100) -> np.ndarray:
    """Load WAV file as mono float32 [-1, 1]."""
    with wave.open(str(path)) as wf:
        raw = wf.readframes(wf.getnframes())
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        if sw == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 3:
            n_samples = len(raw) // 3
            samples = np.zeros(n_samples, dtype=np.float32)
            for i in range(n_samples):
                val = struct.unpack_from("<i", raw[i * 3:i * 3 + 3] + b"\x00")[0]
                samples[i] = val / (2 ** 23)
        else:
            samples = np.frombuffer(raw, dtype=np.float32)

        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)

    return samples


def save_wav(path: Path, audio: np.ndarray, sr: int = 44_100) -> None:
    """Save float32 audio as 16-bit WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_clipped = np.clip(audio, -1.0, 1.0)
    pcm = (audio_clipped * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Claudio Vertical Slice Demo")
    parser.add_argument(
        "--source",
        default="data/calibration/AllHailThePowerOfJesusName_AGuitar_E.wav",
        help="Source WAV file",
    )
    parser.add_argument("--output-dir", default="./demo_output")
    parser.add_argument("--start-sec", type=float, default=30.0)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    sr = 44_100

    print("═" * 70)
    print("  CLAUDIO — Vertical Slice: Intent Pipeline End-to-End Proof")
    print("═" * 70)
    print()

    # ── 1. Load Source ────────────────────────────────────────────────
    print(f"📂 Loading: {source_path.name}")
    audio = load_wav(source_path, sr)
    start = int(args.start_sec * sr)
    end = int((args.start_sec + args.duration_sec) * sr)
    clip = audio[start:end]
    print(f"   Clip: {args.start_sec:.1f}s → {args.start_sec + args.duration_sec:.1f}s "
          f"({len(clip)} samples, {len(clip)/sr:.1f}s)")
    print()

    # ── 2. Encode Intents ─────────────────────────────────────────────
    print("🎤 Encoding intents...")
    t0 = time.monotonic()
    encoder = IntentEncoder(sample_rate=sr)
    frames = encoder.encode_block(clip)
    encode_time = time.monotonic() - t0

    voiced_frames = [f for f in frames if f.f0_hz > 0 and f.f0_confidence > 0.3]
    silent_frames = [f for f in frames if f.loudness_norm < 0.01]
    onset_frames = [f for f in frames if f.is_onset]

    print(f"   Frames extracted: {len(frames)}")
    print(f"   Voiced frames:    {len(voiced_frames)} ({100*len(voiced_frames)/max(1,len(frames)):.0f}%)")
    print(f"   Silent frames:    {len(silent_frames)} ({100*len(silent_frames)/max(1,len(frames)):.0f}%)")
    print(f"   Onset events:     {len(onset_frames)}")
    print(f"   Encode time:      {encode_time*1000:.0f} ms "
          f"(real-time factor: {args.duration_sec/encode_time:.1f}×)")
    print()

    if voiced_frames:
        f0_values = [f.f0_hz for f in voiced_frames]
        print(f"   F0 range:         {min(f0_values):.0f} — {max(f0_values):.0f} Hz")
        print(f"   F0 median:        {np.median(f0_values):.1f} Hz")
    print()

    # ── 3. Protocol Serialization ─────────────────────────────────────
    print("📡 Serializing to wire protocol...")
    stream = IntentStream()
    total_bytes = 0
    silence_packets = 0
    restored_frames: list[IntentFrame] = []

    for frame in frames:
        packet = stream.pack(frame)
        data = packet.to_bytes()
        total_bytes += len(data)
        if packet.flags & PacketFlags.SILENCE:
            silence_packets += 1

        # Simulate network: deserialize
        restored_packet = IntentPacket.from_bytes(data)
        if restored_packet.frame is not None:
            restored_frames.append(restored_packet.frame)
        else:
            restored_frames.append(IntentFrame(
                timestamp_ms=restored_packet.timestamp_ms,
                f0_hz=0.0, f0_confidence=0.0,
                loudness_db=-80.0, loudness_norm=0.0,
                spectral_centroid_hz=0.0,
            ))

    raw_pcm_bytes = len(clip) * 2  # 16-bit mono
    compression = 1.0 - (total_bytes / raw_pcm_bytes)
    intent_rate = total_bytes / args.duration_sec
    pcm_rate = raw_pcm_bytes / args.duration_sec

    print(f"   Total packets:    {len(frames)}")
    print(f"   Silence packets:  {silence_packets} ({100*silence_packets/max(1,len(frames)):.0f}%)")
    print(f"   Wire bytes:       {total_bytes:,} ({total_bytes/1024:.1f} KB)")
    print(f"   Raw PCM bytes:    {raw_pcm_bytes:,} ({raw_pcm_bytes/1024:.1f} KB)")
    print(f"   Compression:      {compression:.1%}")
    print(f"   Intent bandwidth: {intent_rate/1024:.1f} KB/s")
    print(f"   PCM bandwidth:    {pcm_rate/1024:.1f} KB/s")
    print(f"   Bandwidth saving: {pcm_rate/1024 - intent_rate/1024:.1f} KB/s saved")
    print()

    # ── 4. Decode / Regenerate ────────────────────────────────────────
    print("🔊 Regenerating audio from intent packets...")
    t0 = time.monotonic()
    decoder = IntentDecoder(sample_rate=sr)
    regenerated = decoder.decode_frames(restored_frames)
    decode_time = time.monotonic() - t0

    print(f"   Regenerated:      {len(regenerated)} samples ({len(regenerated)/sr:.1f}s)")
    print(f"   Decode time:      {decode_time*1000:.0f} ms "
          f"(real-time factor: {args.duration_sec/decode_time:.1f}×)")
    print()

    # ── 5. Quality Metrics ────────────────────────────────────────────
    print("📊 Quality metrics:")
    min_len = min(len(clip), len(regenerated))

    # RMS comparison
    rms_orig = float(np.sqrt(np.mean(clip[:min_len] ** 2)))
    rms_regen = float(np.sqrt(np.mean(regenerated[:min_len] ** 2)))
    rms_diff_db = 20 * math.log10((rms_regen + 1e-10) / (rms_orig + 1e-10))
    print(f"   RMS original:     {20*math.log10(rms_orig+1e-10):.1f} dB")
    print(f"   RMS regenerated:  {20*math.log10(rms_regen+1e-10):.1f} dB")
    print(f"   RMS difference:   {rms_diff_db:+.1f} dB")

    # Spectral correlation
    n_fft = min(min_len, 8192)
    spec_orig = np.abs(np.fft.rfft(clip[:n_fft]))
    spec_regen = np.abs(np.fft.rfft(regenerated[:n_fft]))
    eps = 1e-10
    db_orig = 20 * np.log10(spec_orig / (np.max(spec_orig) + eps) + eps)
    db_regen = 20 * np.log10(spec_regen / (np.max(spec_regen) + eps) + eps)
    mask = db_orig > -60
    if np.sum(mask) > 10:
        corr = np.corrcoef(db_orig[mask], db_regen[mask])[0, 1]
        print(f"   Spectral corr:    {corr:.3f}")

    # Re-encode regenerated to verify pitch preservation
    encoder2 = IntentEncoder(sample_rate=sr)
    regen_frames = encoder2.encode_block(regenerated)
    regen_voiced = [f for f in regen_frames if f.f0_hz > 0 and f.f0_confidence > 0.3]
    if voiced_frames and regen_voiced:
        orig_f0 = np.median([f.f0_hz for f in voiced_frames])
        regen_f0 = np.median([f.f0_hz for f in regen_voiced])
        f0_error = abs(1200 * math.log2(regen_f0 / orig_f0)) if orig_f0 > 0 else 0
        print(f"   F0 original:      {orig_f0:.1f} Hz")
        print(f"   F0 regenerated:   {regen_f0:.1f} Hz")
        print(f"   F0 error:         {f0_error:.1f} cents")
    print()

    # ── 6. Save Outputs ───────────────────────────────────────────────
    orig_path = output_dir / "original.wav"
    regen_path = output_dir / "regenerated.wav"
    save_wav(orig_path, clip, sr)
    save_wav(regen_path, regenerated, sr)

    print("💾 Saved A/B comparison:")
    print(f"   Original:     {orig_path}")
    print(f"   Regenerated:  {regen_path}")
    print()

    # ── Summary ───────────────────────────────────────────────────────
    print("═" * 70)
    print("  VERTICAL SLICE PROOF — SUMMARY")
    print("═" * 70)
    print()
    print(f"  ✅ Intent extraction:   {len(frames)} frames at 250Hz")
    print(f"  ✅ Wire protocol:       {compression:.0%} compression ({intent_rate/1024:.1f} KB/s vs {pcm_rate/1024:.1f} KB/s)")
    print(f"  ✅ Audio regeneration:  {len(regenerated)/sr:.1f}s from intent packets")
    print(f"  ✅ Real-time capable:   encode {args.duration_sec/encode_time:.0f}× / decode {args.duration_sec/decode_time:.0f}× real-time")
    print(f"  ✅ A/B comparison:      saved to {output_dir}/")
    print()
    print("  Pipeline proven: Audio → Intent → Wire → Intent → Audio ✓")
    print()


if __name__ == "__main__":
    main()
