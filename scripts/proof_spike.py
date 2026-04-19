"""
proof_spike.py — Prove the gap and the fix.

1. Load a real guitar recording
2. Run it through Intent Encoder → Intent Decoder (additive synth)
3. Save the output as "regen_additive.wav"
4. Run it through EnCodec encode → decode
5. Save the output as "regen_encodec.wav"
6. Print bandwidth comparison
"""

import os
import struct
import time
import wave

import numpy as np

from src.claudio.intent.intent_decoder import IntentDecoder
from src.claudio.intent.intent_encoder import IntentEncoder
from src.claudio.intent.intent_protocol import IntentStream

# Add project to path

OUTPUT_DIR = "demo_output/proof"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as float32 mono."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

    if sample_width == 2:
        fmt = f"<{n_frames * channels}h"
        samples = struct.unpack(fmt, raw)
        audio = np.array(samples, dtype=np.float32) / 32768.0
    elif sample_width == 4:
        fmt = f"<{n_frames * channels}f"
        samples = struct.unpack(fmt, raw)
        audio = np.array(samples, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Mix to mono if stereo
    if channels == 2:
        audio = (audio[0::2] + audio[1::2]) / 2.0

    return audio, sr


def save_wav(path: str, audio: np.ndarray, sr: int):
    """Save float32 audio as 16-bit WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def rms_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute RMS error in dB."""
    min_len = min(len(original), len(reconstructed))
    error = original[:min_len] - reconstructed[:min_len]
    rms = np.sqrt(np.mean(error**2) + 1e-10)
    return 20 * np.log10(rms)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Current Additive Synth (the lie)
# ═══════════════════════════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════════════════╗")
print("║  CLAUDIO PROOF SPIKE — Additive Synth vs EnCodec           ║")
print("╚══════════════════════════════════════════════════════════════╝")

FIXTURE = "tests/audio_fixtures/electric_guitar.wav"
print(f"\n  Loading: {FIXTURE}")
audio, sr = load_wav(FIXTURE)
print(f"  Sample rate: {sr} Hz, Duration: {len(audio) / sr:.2f}s, Samples: {len(audio)}")

# Use 2 seconds max for the test
max_samples = sr * 2
audio = audio[:max_samples]

print("\n  ── TEST 1: Intent Pipeline (Additive Synth) ──")

encoder = IntentEncoder(sample_rate=sr)
decoder = IntentDecoder(sample_rate=sr)
stream = IntentStream()

t0 = time.perf_counter()
frames = encoder.encode_block(audio)
t_encode = time.perf_counter() - t0

# Measure wire bandwidth
total_bytes = 0
for f in frames:
    pkt = stream.pack(f)
    total_bytes += len(pkt.to_bytes())

t0 = time.perf_counter()
regen = decoder.decode_frames(frames)
t_decode = time.perf_counter() - t0

save_wav(f"{OUTPUT_DIR}/original.wav", audio, sr)
save_wav(f"{OUTPUT_DIR}/regen_additive.wav", regen, sr)

duration = len(audio) / sr
bw_kbps = (total_bytes * 8) / duration / 1000

print(f"  Frames extracted: {len(frames)}")
print(f"  Encode time: {t_encode * 1000:.1f}ms")
print(f"  Decode time: {t_decode * 1000:.1f}ms")
print(f"  Wire bandwidth: {total_bytes} bytes = {bw_kbps:.1f} kbps")
print(f"  RMS error: {rms_error(audio, regen):.1f} dBFS")
print(f"  Output: {OUTPUT_DIR}/regen_additive.wav")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: EnCodec (the fix)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n  ── TEST 2: EnCodec Neural Codec ──")

try:
    import torch
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    # Load 24kHz model (best quality)
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)  # 6 kbps — lowest quality tier
    model.eval()

    # Convert audio to 24kHz mono for EnCodec
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    # Resample if needed
    if sr != 24000:
        audio_24k = convert_audio(audio_tensor, sr, model.sample_rate, model.channels)
    else:
        audio_24k = audio_tensor

    t0 = time.perf_counter()
    with torch.no_grad():
        encoded = model.encode(audio_24k)
    t_encode_ec = time.perf_counter() - t0

    # Measure codec bandwidth
    codec_bytes = sum(frame.shape[-1] * frame.shape[-2] * 2 for frame in encoded[0])  # int16 codes

    t0 = time.perf_counter()
    with torch.no_grad():
        decoded = model.decode(encoded)
    t_decode_ec = time.perf_counter() - t0

    regen_ec = decoded.squeeze().cpu().numpy()
    # Resample back to original SR for comparison
    if sr != 24000:
        import torchaudio

        regen_ec_tensor = torch.from_numpy(regen_ec).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(24000, sr)
        regen_ec = resampler(regen_ec_tensor).squeeze().numpy()

    save_wav(f"{OUTPUT_DIR}/regen_encodec_6kbps.wav", regen_ec, sr)

    ec_bw_kbps = (codec_bytes * 8) / duration / 1000

    print(f"  EnCodec bandwidth: {codec_bytes} bytes = {ec_bw_kbps:.1f} kbps (target: 6 kbps)")
    print(f"  Encode time: {t_encode_ec * 1000:.1f}ms")
    print(f"  Decode time: {t_decode_ec * 1000:.1f}ms")
    print(f"  RMS error: {rms_error(audio, regen_ec):.1f} dBFS")
    print(f"  Output: {OUTPUT_DIR}/regen_encodec_6kbps.wav")

    # Also test at 24 kbps (high quality)
    model.set_target_bandwidth(24.0)
    with torch.no_grad():
        encoded_hq = model.encode(audio_24k)
        decoded_hq = model.decode(encoded_hq)
    regen_hq = decoded_hq.squeeze().cpu().numpy()
    if sr != 24000:
        regen_hq_tensor = torch.from_numpy(regen_hq).float().unsqueeze(0)
        regen_hq = resampler(regen_hq_tensor).squeeze().numpy()
    save_wav(f"{OUTPUT_DIR}/regen_encodec_24kbps.wav", regen_hq, sr)

    codec_bytes_hq = sum(frame.shape[-1] * frame.shape[-2] * 2 for frame in encoded_hq[0])
    ec_bw_hq = (codec_bytes_hq * 8) / duration / 1000
    print(f"\n  EnCodec 24kbps RMS error: {rms_error(audio, regen_hq):.1f} dBFS")
    print(f"  Output: {OUTPUT_DIR}/regen_encodec_24kbps.wav")

except ImportError as e:
    print(f"  ⚠ EnCodec not installed: {e}")
    print("  Run: pip install encodec")
    codec_bytes = 0
    ec_bw_kbps = 0
    ec_bw_hq = 0

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

print("\n  ═══════════════════════════════════════════════════")
print("  BANDWIDTH COMPARISON")
print("  ═══════════════════════════════════════════════════")
raw_bw = sr * 2 * 8 / 1000  # 16-bit mono PCM
print(f"  Raw PCM (16-bit mono):     {raw_bw:.0f} kbps")
print(f"  Intent packets (additive): {bw_kbps:.1f} kbps  ({raw_bw / bw_kbps:.0f}× compression)")
if ec_bw_kbps > 0:
    print(f"  EnCodec @ 6 kbps:          {ec_bw_kbps:.1f} kbps  ({raw_bw / ec_bw_kbps:.0f}× compression)")
    print(f"  EnCodec @ 24 kbps:         {ec_bw_hq:.1f} kbps ({raw_bw / ec_bw_hq:.0f}× compression)")
print(f"\n  ⚡ Hybrid (intent + encodec 6k): ~{bw_kbps + 6:.0f} kbps total")
print(f"     Still {raw_bw / (bw_kbps + 6):.0f}× less than raw PCM")
print("  ═══════════════════════════════════════════════════")
