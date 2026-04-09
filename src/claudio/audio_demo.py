"""
audio_demo.py — Claudio Audio Processing Demo Generator

Generates realistic instrument audio using additive synthesis with
proper harmonics, envelopes, and timbral characteristics, then
processes each through the HRTF binaural engine to produce
before/after WAV files for audible comparison.

Instruments synthesised:
  - Acoustic guitar (nylon string — rich harmonics, soft attack)
  - Electric bass (fingerstyle — strong fundamental, warm tone)
  - Piano (grand — bright attack, sustained harmonics)
  - Drums (kick + snare + hi-hat pattern)

Each instrument is rendered at multiple spatial positions to
demonstrate the binaural spatialisation effect.

Usage:
    python -m claudio.audio_demo
    # Outputs WAV files to demo_output/
"""

from __future__ import annotations

import math
import wave

import numpy as np

from claudio.hrtf_engine import AudioSource, HRTFBinauralEngine
from claudio.signal_flow_config import SignalFlowConfig

DEMO_SAMPLE_RATE = 48_000
DEMO_DIR = "demo_output"


# ─── Instrument Synthesis ────────────────────────────────────────────────────


def _adsr_envelope(
    n_samples: int,
    sr: int,
    attack_s: float = 0.01,
    decay_s: float = 0.1,
    sustain_level: float = 0.7,
    release_s: float = 0.3,
) -> np.ndarray:
    """ADSR envelope generator."""
    a = min(int(attack_s * sr), n_samples)
    d = min(int(decay_s * sr), n_samples - a)
    r = min(int(release_s * sr), n_samples - a - d)
    s = n_samples - a - d - r

    env = np.zeros(n_samples, dtype=np.float64)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if d > 0:
        env[a : a + d] = np.linspace(1, sustain_level, d)
    if s > 0:
        env[a + d : a + d + s] = sustain_level
    if r > 0:
        env[a + d + s : a + d + s + r] = np.linspace(sustain_level, 0, r)
    return env


def synthesise_guitar_note(
    freq: float,
    duration_s: float,
    sr: int = DEMO_SAMPLE_RATE,
) -> np.ndarray:
    """Acoustic guitar — Karplus-Strong-inspired with harmonic series."""
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr

    # Harmonic series with guitar-like spectral envelope
    harmonics = [1.0, 0.8, 0.5, 0.35, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03]
    signal = np.zeros(n, dtype=np.float64)
    for i, amp in enumerate(harmonics):
        h = i + 1
        # Slight inharmonicity (string stiffness)
        f_h = freq * h * (1 + 0.0003 * h * h)
        # Higher harmonics decay faster
        decay = math.exp(-0.5 * h)
        signal += amp * decay * np.sin(2 * np.pi * f_h * t) * np.exp(-3.0 * h * t)

    env = _adsr_envelope(n, sr, attack_s=0.005, decay_s=0.15, sustain_level=0.4, release_s=0.4)
    return (signal * env * 0.6).astype(np.float32)


def synthesise_bass_note(
    freq: float,
    duration_s: float,
    sr: int = DEMO_SAMPLE_RATE,
) -> np.ndarray:
    """Electric bass — strong fundamental, warm upper harmonics."""
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr

    harmonics = [1.0, 0.6, 0.3, 0.15, 0.08, 0.04]
    signal = np.zeros(n, dtype=np.float64)
    for i, amp in enumerate(harmonics):
        h = i + 1
        signal += amp * np.sin(2 * np.pi * freq * h * t) * np.exp(-1.5 * h * t)

    env = _adsr_envelope(n, sr, attack_s=0.01, decay_s=0.08, sustain_level=0.6, release_s=0.3)
    return (signal * env * 0.45).astype(np.float32)


def synthesise_piano_note(
    freq: float,
    duration_s: float,
    sr: int = DEMO_SAMPLE_RATE,
) -> np.ndarray:
    """Grand piano — bright attack with sustained ringing harmonics."""
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr

    harmonics = [1.0, 0.9, 0.55, 0.4, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01]
    signal = np.zeros(n, dtype=np.float64)
    for i, amp in enumerate(harmonics):
        h = i + 1
        f_h = freq * h * (1 + 0.0005 * h * h)  # piano inharmonicity
        signal += amp * np.sin(2 * np.pi * f_h * t) * np.exp(-0.8 * h * t)

    env = _adsr_envelope(n, sr, attack_s=0.002, decay_s=0.2, sustain_level=0.5, release_s=0.8)
    return (signal * env * 0.35).astype(np.float32)


def synthesise_drum_pattern(
    duration_s: float,
    bpm: float = 120.0,
    sr: int = DEMO_SAMPLE_RATE,
) -> np.ndarray:
    """Drum pattern — kick, snare, hi-hat."""
    n = int(sr * duration_s)
    signal = np.zeros(n, dtype=np.float64)
    beat_samples = int(60.0 / bpm * sr)

    for beat in range(int(duration_s * bpm / 60)):
        offset = beat * beat_samples
        if offset >= n:
            break

        # Kick on beats 1 and 3
        if beat % 4 in (0, 2):
            kick_len = min(int(0.15 * sr), n - offset)
            t = np.arange(kick_len, dtype=np.float64) / sr
            kick_freq = 60 * np.exp(-15 * t)  # pitch sweep down
            kick = 0.8 * np.sin(2 * np.pi * np.cumsum(kick_freq) / sr) * np.exp(-12 * t)
            signal[offset : offset + kick_len] += kick

        # Snare on beats 2 and 4
        if beat % 4 in (1, 3):
            snare_len = min(int(0.12 * sr), n - offset)
            t = np.arange(snare_len, dtype=np.float64) / sr
            rng = np.random.default_rng(seed=beat)
            noise = rng.standard_normal(snare_len) * 0.3 * np.exp(-15 * t)
            tone = 0.4 * np.sin(2 * np.pi * 200 * t) * np.exp(-20 * t)
            signal[offset : offset + snare_len] += noise + tone

        # Hi-hat on every 8th note
        hh_offset = offset + (beat_samples // 2 if beat % 2 == 1 else 0)
        hh_len = min(int(0.04 * sr), n - hh_offset)
        if hh_len > 0 and hh_offset < n:
            t = np.arange(hh_len, dtype=np.float64) / sr
            rng = np.random.default_rng(seed=beat + 1000)
            hh = rng.standard_normal(hh_len) * 0.15 * np.exp(-40 * t)
            end = min(hh_offset + hh_len, n)
            signal[hh_offset:end] += hh[: end - hh_offset]

    return (signal * 0.6).astype(np.float32)


# ─── Melody / Phrase Generators ──────────────────────────────────────────────


def _guitar_phrase(sr: int = DEMO_SAMPLE_RATE) -> np.ndarray:
    """Am pentatonic melodic phrase — ~4 seconds."""
    notes = [
        (220.0, 0.4),
        (261.6, 0.4),
        (293.7, 0.3),
        (329.6, 0.5),
        (392.0, 0.3),
        (329.6, 0.4),
        (293.7, 0.3),
        (261.6, 0.5),
        (220.0, 0.8),
    ]
    parts = [synthesise_guitar_note(f, d, sr) for f, d in notes]
    return np.concatenate(parts)


def _bass_line(sr: int = DEMO_SAMPLE_RATE) -> np.ndarray:
    """Walking bass line — ~4 seconds."""
    notes = [
        (82.4, 0.5),
        (98.0, 0.5),
        (110.0, 0.5),
        (98.0, 0.5),
        (82.4, 0.5),
        (73.4, 0.5),
        (82.4, 0.5),
        (110.0, 0.5),
    ]
    parts = [synthesise_bass_note(f, d, sr) for f, d in notes]
    return np.concatenate(parts)


def _piano_chord_progression(sr: int = DEMO_SAMPLE_RATE) -> np.ndarray:
    """Am - F - C - G progression — ~4 seconds."""
    chords = [
        ([220.0, 261.6, 329.6], 1.0),  # Am
        ([174.6, 220.0, 261.6], 1.0),  # F
        ([261.6, 329.6, 392.0], 1.0),  # C
        ([196.0, 246.9, 293.7], 1.0),  # G
    ]
    parts = []
    for freqs, dur in chords:
        chord = sum(synthesise_piano_note(f, dur, sr) for f in freqs)
        parts.append(chord / len(freqs))
    return np.concatenate(parts)


# ─── WAV I/O ─────────────────────────────────────────────────────────────────


def write_wav_mono(filepath: str, audio: np.ndarray, sr: int) -> None:
    """Write 16-bit mono WAV file."""
    audio_16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_16.tobytes())


def write_wav_stereo(filepath: str, left: np.ndarray, right: np.ndarray, sr: int) -> None:
    """Write 16-bit stereo WAV file."""
    n = min(len(left), len(right))
    left_16 = np.clip(left[:n] * 32767, -32768, 32767).astype(np.int16)
    right_16 = np.clip(right[:n] * 32767, -32768, 32767).astype(np.int16)
    interleaved = np.empty(n * 2, dtype=np.int16)
    interleaved[0::2] = left_16
    interleaved[1::2] = right_16
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())


# ─── Processing Pipeline ────────────────────────────────────────────────────


def process_through_claudio(
    mono_audio: np.ndarray,
    source_position: np.ndarray,
    head_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    label: str = "src",
) -> tuple[np.ndarray, np.ndarray]:
    """Run mono audio through the HRTF binaural engine → stereo output."""
    cfg = SignalFlowConfig(
        capture_sample_rate=DEMO_SAMPLE_RATE,
        render_sample_rate=DEMO_SAMPLE_RATE,  # No resampling for demo
        fft_size=512,
        hrir_length=256,
    )
    engine = HRTFBinauralEngine(config=cfg)
    src = AudioSource(source_id=label, position=source_position)
    engine.add_source(src)
    engine.update_head_pose(head_quat)

    block = cfg.fft_size
    n_blocks = len(mono_audio) // block
    out_l_parts, out_r_parts = [], []

    for b in range(n_blocks):
        chunk = mono_audio[b * block : (b + 1) * block]
        frame = engine.render({label: chunk})
        out_l_parts.append(frame.left)
        out_r_parts.append(frame.right)

    out_l = np.concatenate(out_l_parts) if out_l_parts else np.zeros(0, dtype=np.float32)
    out_r = np.concatenate(out_r_parts) if out_r_parts else np.zeros(0, dtype=np.float32)

    # Normalise to prevent clipping
    peak = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-6)
    if peak > 0.95:
        out_l *= 0.9 / peak
        out_r *= 0.9 / peak

    return out_l, out_r
