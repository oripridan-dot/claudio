"""
generate_ab_demo.py — Generate A/B Demo WAV Files

Creates high-fidelity dry (mono) and Claudio-processed (binaural stereo)
WAV files for side-by-side listening comparison.

All audio generated at 48kHz/32-bit float for maximum browser compatibility.
Claudio internally renders at 192kHz then the engine outputs at render rate,
which we downsample back to 48kHz for the WAV export.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from claudio.hrtf_engine import AudioSource, HRTFBinauralEngine
from claudio.intent.intent_decoder import IntentDecoder
from claudio.intent.intent_encoder import IntentEncoder
from claudio.signal_flow_config import balanced_config

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "demo_output" / "ab_demo"
SR = 48_000  # export sample rate


# ═══════════════════════════════════════════════════════════════════════
# High-Fidelity Audio Synthesis
# ═══════════════════════════════════════════════════════════════════════

def guitar_chord(dur: float = 3.0) -> np.ndarray:
    """Rich acoustic guitar chord (Am7) via Karplus-Strong with body resonance."""
    n = int(SR * dur)
    mix = np.zeros(n, dtype=np.float64)
    # Am7 chord: A2, C3, E3, G3, A3
    freqs = [110.0, 130.81, 164.81, 196.0, 220.0]
    delays_ms = [0, 12, 24, 36, 48]  # strum delay

    for freq, delay_ms in zip(freqs, delays_ms, strict=False):
        period = max(2, int(SR / freq))
        rng = np.random.default_rng(int(freq * 100))
        buf = rng.uniform(-0.5, 0.5, period).astype(np.float64)
        # Low-pass the initial excitation for warmer tone
        for _j in range(3):
            buf_new = np.copy(buf)
            for k in range(1, period - 1):
                buf_new[k] = 0.25 * buf[k - 1] + 0.5 * buf[k] + 0.25 * buf[k + 1]
            buf = buf_new

        start = int(delay_ms / 1000 * SR)
        string_out = np.zeros(n, dtype=np.float64)
        for i in range(n - start):
            idx = i % period
            string_out[i + start] = buf[idx]
            # Damped averaging with slight detuning
            nxt = (idx + 1) % period
            buf[idx] = 0.998 * (0.5 * buf[idx] + 0.5 * buf[nxt])

        mix += string_out * 0.35

    # Body resonance: add subtle formant at ~100Hz and ~250Hz
    t = np.arange(n, dtype=np.float64) / SR
    body = 0.03 * np.sin(2 * np.pi * 100 * t) * np.exp(-t * 3)
    body += 0.02 * np.sin(2 * np.pi * 250 * t) * np.exp(-t * 4)
    mix += body * np.clip(np.abs(mix) * 10, 0, 1)  # only when signal present

    mx = np.max(np.abs(mix)) + 1e-10
    return (mix / mx * 0.85).astype(np.float32)


def piano_melody(dur: float = 4.0) -> np.ndarray:
    """Realistic piano melody with proper hammer strike and sustain."""
    n = int(SR * dur)
    mix = np.zeros(n, dtype=np.float64)
    np.arange(n, dtype=np.float64) / SR

    # C major arpeggio: C4, E4, G4, C5, then resolve to C4
    notes = [
        (261.63, 0.0, 1.5),   # C4
        (329.63, 0.4, 1.2),   # E4
        (392.00, 0.8, 1.0),   # G4
        (523.25, 1.2, 0.8),   # C5
        (493.88, 1.8, 1.0),   # B4
        (440.00, 2.2, 1.0),   # A4
        (392.00, 2.6, 0.8),   # G4
        (261.63, 3.0, 1.0),   # C4 resolve
    ]

    for f0, onset_s, note_dur in notes:
        onset = int(onset_s * SR)
        note_n = min(int(note_dur * SR), n - onset)
        if note_n <= 0:
            continue
        t = np.arange(note_n, dtype=np.float64) / SR

        sig = np.zeros(note_n, dtype=np.float64)
        B = 0.00015  # inharmonicity

        for h in range(1, 20):
            f_h = f0 * h * math.sqrt(1 + B * h * h)
            if f_h > SR / 2:
                break
            # Amplitude: higher harmonics decay faster
            amp = 1.0 / (h ** 0.65)
            # Hammer position affects odd/even balance
            hammer_pos = 0.12  # hammer strikes at 12% of string length
            hammer_weight = abs(math.sin(math.pi * h * hammer_pos))
            amp *= hammer_weight
            # Per-partial decay: higher partials decay faster
            decay_rate = 0.8 + h * 0.25
            env = np.exp(-t * decay_rate)
            # Attack transient
            attack = 1 - np.exp(-t * 200)
            sig += amp * np.sin(2 * np.pi * f_h * t) * env * attack

        # Velocity scaling
        vel = 0.7 + 0.3 * math.sin(onset_s * 1.5)
        sig *= vel

        mx_note = np.max(np.abs(sig)) + 1e-10
        mix[onset:onset + note_n] += sig / mx_note * 0.6

    mx = np.max(np.abs(mix)) + 1e-10
    return (mix / mx * 0.85).astype(np.float32)


def full_band_mix(dur: float = 4.0) -> np.ndarray:
    """Full band mix: bass, keys, lead, drums — simulated mastered track."""
    n = int(SR * dur)
    np.arange(n, dtype=np.float64) / SR
    mix = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(42)

    # ── Bass (sub + harmonics with slide) ───
    bass_notes = [(82.41, 0.0), (73.42, 1.0), (98.0, 2.0), (82.41, 3.0)]
    for f0, onset in bass_notes:
        start = int(onset * SR)
        seg_dur = min(1.0, dur - onset)
        seg_n = int(seg_dur * SR)
        seg_t = np.arange(seg_n, dtype=np.float64) / SR
        env = (1 - np.exp(-seg_t * 30)) * np.exp(-seg_t * 2)
        bass = np.zeros(seg_n, dtype=np.float64)
        for h in range(1, 5):
            bass += (0.8 / h) * np.sin(2 * np.pi * f0 * h * seg_t) * env
        mix[start:start + seg_n] += bass * 0.35

    # ── Electric piano / keys ───
    key_notes = [
        (329.63, 0.0, 0.8), (349.23, 0.5, 0.6), (392.00, 1.0, 0.8),
        (440.00, 1.5, 0.6), (493.88, 2.0, 0.8), (440.00, 2.5, 0.6),
        (392.00, 3.0, 0.8), (329.63, 3.5, 0.5),
    ]
    for f0, onset, nd in key_notes:
        start = int(onset * SR)
        seg_n = min(int(nd * SR), n - start)
        if seg_n <= 0:
            continue
        seg_t = np.arange(seg_n, dtype=np.float64) / SR
        env = np.exp(-seg_t * 3) * (1 - np.exp(-seg_t * 80))
        keys = np.zeros(seg_n, dtype=np.float64)
        for h in [1, 2, 3, 4, 6]:
            keys += (0.4 / h) * np.sin(2 * np.pi * f0 * h * seg_t)
        # FM shimmer
        mod = 1.0 + 0.02 * np.sin(2 * np.pi * 5.5 * seg_t)
        keys *= mod
        mix[start:start + seg_n] += keys * env * 0.2

    # ── Lead melody (sawtooth-ish with vibrato) ───
    lead_notes = [
        (523.25, 0.5, 0.4), (587.33, 1.0, 0.4), (659.26, 1.5, 0.6),
        (587.33, 2.2, 0.3), (523.25, 2.6, 0.5), (440.00, 3.2, 0.8),
    ]
    for f0, onset, nd in lead_notes:
        start = int(onset * SR)
        seg_n = min(int(nd * SR), n - start)
        if seg_n <= 0:
            continue
        seg_t = np.arange(seg_n, dtype=np.float64) / SR
        vibrato = f0 * (1 + 0.005 * np.sin(2 * np.pi * 5 * seg_t))
        phase = 2 * np.pi * np.cumsum(vibrato) / SR
        lead = np.zeros(seg_n, dtype=np.float64)
        for h in range(1, 10):
            lead += (0.5 / h) * np.sin(h * phase) * ((-1) ** (h + 1))
        env = np.exp(-seg_t * 2) * (1 - np.exp(-seg_t * 60))
        mix[start:start + seg_n] += lead * env * 0.15

    # ── Drums ───
    bpm = 120
    beat_dur = 60.0 / bpm
    for beat in range(int(dur / beat_dur)):
        pos = int(beat * beat_dur * SR)
        # Kick on 1, 3
        if beat % 2 == 0:
            kick_n = min(int(0.15 * SR), n - pos)
            if kick_n > 0:
                kick_t = np.arange(kick_n, dtype=np.float64) / SR
                kick_env = np.exp(-kick_t * 25)
                kick_pitch = 55 + 100 * np.exp(-kick_t * 40)
                kick = np.sin(2 * np.pi * np.cumsum(kick_pitch / SR)) * kick_env
                mix[pos:pos + kick_n] += kick * 0.4
        # Snare on 2, 4
        if beat % 2 == 1:
            snr_n = min(int(0.12 * SR), n - pos)
            if snr_n > 0:
                snr_t = np.arange(snr_n, dtype=np.float64) / SR
                snare = (rng.normal(0, 1, snr_n) * np.exp(-snr_t * 18) * 0.3
                         + np.sin(2 * np.pi * 200 * snr_t) * np.exp(-snr_t * 30) * 0.5)
                mix[pos:pos + snr_n] += snare * 0.3
        # Hi-hat on every beat
        hh_n = min(int(0.04 * SR), n - pos)
        if hh_n > 0:
            hh_t = np.arange(hh_n, dtype=np.float64) / SR
            hh = rng.normal(0, 1, hh_n) * np.exp(-hh_t * 60)
            mix[pos:pos + hh_n] += hh * 0.12

    mx = np.max(np.abs(mix)) + 1e-10
    return (mix / mx * 0.85).astype(np.float32)


def orchestral_swell(dur: float = 4.0) -> np.ndarray:
    """Orchestral string swell — rich harmonic content with crescendo."""
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float64) / SR
    mix = np.zeros(n, dtype=np.float64)

    # String ensemble: C3, G3, E4, G4, C5 (open voicing)
    chord_freqs = [130.81, 196.0, 329.63, 392.0, 523.25]

    # Crescendo envelope: slow build → sustain → gentle release
    env = np.clip(t / 2.0, 0, 1) * np.exp(-np.clip(t - 3.0, 0, None) * 2)

    for i, f0 in enumerate(chord_freqs):
        # Each string with slight detuning for ensemble effect
        for detune in [-0.3, 0.0, 0.3]:
            freq = f0 * (1 + detune / 100)
            # Vibrato (different rate per voice for realism)
            vib_rate = 5.0 + i * 0.3
            vibrato = freq * (1 + 0.004 * np.sin(2 * np.pi * vib_rate * t))
            phase = 2 * np.pi * np.cumsum(vibrato) / SR

            sig = np.zeros(n, dtype=np.float64)
            # Rich sawtooth-like timbre (bowed string)
            for h in range(1, 12):
                amp = 1.0 / h
                sig += amp * np.sin(h * phase)

            mix += sig * env * 0.08

    mx = np.max(np.abs(mix)) + 1e-10
    return (mix / mx * 0.85).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# WAV Export (32-bit float)
# ═══════════════════════════════════════════════════════════════════════

def write_wav_float32(path: Path, data: np.ndarray, sr: int, channels: int = 1):
    """Write 32-bit float WAV file (IEEE float format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = data.astype(np.float32)

    n_frames = len(data) // channels
    data_bytes = data.tobytes()

    with open(path, "wb") as f:
        # RIFF header
        data_size = len(data_bytes)
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk (IEEE float = format 3)
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))      # chunk size
        f.write(struct.pack("<H", 3))       # IEEE float
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", sr * channels * 4))  # byte rate
        f.write(struct.pack("<H", channels * 4))       # block align
        f.write(struct.pack("<H", 32))      # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data_bytes)

    print(f"  📁 {path.name} ({data_size / 1024:.0f} KB, {n_frames / sr:.1f}s, {channels}ch)")


# ═══════════════════════════════════════════════════════════════════════
# Claudio HRTF Processing
# ═══════════════════════════════════════════════════════════════════════

def process_binaural(
    mono: np.ndarray, azimuth: float = 45.0, elevation: float = 0.0,
) -> np.ndarray:
    """Process mono audio through Claudio HRTF → stereo binaural."""
    config = balanced_config()
    engine = HRTFBinauralEngine(config=config)
    block = config.fft_size

    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)
    pos = np.array([
        2.0 * math.sin(az_rad) * math.cos(el_rad),
        2.0 * math.sin(el_rad),
        -2.0 * math.cos(az_rad) * math.cos(el_rad),
    ])

    src = AudioSource(source_id="demo", position=pos)
    engine.add_source(src)

    # Upsample to render rate
    ratio = config.render_sample_rate // SR
    data = np.repeat(mono, ratio) if ratio > 1 else mono.copy()

    n_blocks = max(1, len(data) // block)
    out_l_list, out_r_list = [], []

    for b in range(n_blocks):
        chunk = data[b * block:(b + 1) * block]
        if len(chunk) < block:
            chunk = np.pad(chunk, (0, block - len(chunk)))
        frame = engine.render({"demo": chunk})
        out_l_list.append(frame.left)
        out_r_list.append(frame.right)

    out_l = np.concatenate(out_l_list)
    out_r = np.concatenate(out_r_list)

    # Downsample back to export rate
    if ratio > 1:
        out_l = out_l[::ratio]
        out_r = out_r[::ratio]

    # Trim to original length
    orig_len = len(mono)
    out_l = out_l[:orig_len]
    out_r = out_r[:orig_len]

    # Interleave L/R for stereo WAV
    stereo = np.empty(len(out_l) + len(out_r), dtype=np.float32)
    stereo[0::2] = out_l
    stereo[1::2] = out_r
    return stereo


def process_intent_resynthesis(mono: np.ndarray) -> np.ndarray:
    """Process mono audio through Intent Encoder -> Decoder for resynthesis preview."""
    encoder = IntentEncoder(sample_rate=SR)
    decoder = IntentDecoder(sample_rate=SR, frame_rate=250, n_harmonics=40)
    hop = encoder.hop
    n_blocks = len(mono) // hop

    out_audio = np.zeros(len(mono), dtype=np.float32)

    for i in range(n_blocks):
        start = i * hop
        block = mono[start : start + encoder.frame_len]
        if len(block) < encoder.frame_len:
            block = np.pad(block, (0, encoder.frame_len - len(block)))

        frames = encoder.encode_block(block, start_time_ms=(start/SR)*1000)

        if frames:
            frame = frames[0]
            decoded_chunk = decoder._decode_single_frame(frame)
            out_audio[start : start + hop] = decoded_chunk[:hop]

    return out_audio


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

DEMO_SAMPLES = [
    ("guitar_chord", "Acoustic Guitar — Am7 Chord", guitar_chord, 45.0),
    ("piano_melody", "Piano — C Major Arpeggio", piano_melody, -30.0),
    ("full_band", "Full Band Mix — Bass, Keys, Lead, Drums", full_band_mix, 60.0),
    ("orchestral", "Orchestral String Swell", orchestral_swell, -45.0),
]


def main():
    print("═" * 60)
    print("  CLAUDIO A/B DEMO — Generating High-Fidelity Audio")
    print("═" * 60)
    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Format: 48kHz / 32-bit float WAV")
    print("  Engine: 192kHz internal render, bilinear HRTF\n")

    for slug, name, gen_fn, azimuth in DEMO_SAMPLES:
        print(f"▶ {name}")

        # Generate dry mono
        mono = gen_fn()
        dry_path = OUTPUT_DIR / f"{slug}_dry.wav"
        write_wav_float32(dry_path, mono, SR, channels=1)

        # Process through Claudio
        stereo = process_binaural(mono, azimuth=azimuth)
        wet_path = OUTPUT_DIR / f"{slug}_claudio.wav"
        write_wav_float32(wet_path, stereo, SR, channels=2)

        # Process through Intent Resynthesis
        resyn = process_intent_resynthesis(mono)
        resyn_path = OUTPUT_DIR / f"{slug}_intent.wav"
        write_wav_float32(resyn_path, resyn, SR, channels=1)

        print()

    print("✅ All demos generated. Use headphones for binaural effect!")
    print(f"   Files: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
