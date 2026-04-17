#!/usr/bin/env python3
"""
generate_test_audio.py — Generate Realistic Instrument Test Audio

Creates high-quality synthetic instrument recordings using physics-based
sound synthesis. These are far more realistic than simple sine waves
because they model:
  - Physical string/membrane vibrations (Karplus-Strong, modal synthesis)
  - Body resonances and radiation patterns
  - Realistic attack/sustain/release envelopes
  - Time-varying spectra (formants, vibrato, pitch glides)
  - Room ambience and noise floor

Output: tests/audio_fixtures/ directory with labeled WAV files
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SR = 48_000
DURATION = 3.0
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "audio_fixtures"


def karplus_strong(freq: float, duration: float, sr: int, brightness: float = 0.5) -> np.ndarray:
    """Karplus-Strong plucked string synthesis — physically accurate."""
    n_samples = int(sr * duration)
    delay_len = int(sr / freq)
    if delay_len < 2:
        delay_len = 2

    # Initial excitation — filtered noise burst
    buf = np.random.randn(delay_len).astype(np.float64)
    # Brightness filter on excitation
    for i in range(1, len(buf)):
        buf[i] = brightness * buf[i] + (1 - brightness) * buf[i - 1]

    output = np.zeros(n_samples, dtype=np.float64)
    output[:delay_len] = buf

    # String loop with loss filter
    loss = 0.996  # decay factor
    for i in range(delay_len, n_samples):
        avg = 0.5 * (output[i - delay_len] + output[i - delay_len + 1])
        output[i] = loss * avg

    return output.astype(np.float32)


def generate_electric_guitar() -> np.ndarray:
    """Electric guitar — Karplus-Strong with distortion and pickup simulation."""
    notes = [329.63, 246.94, 196.0, 329.63, 392.0, 329.63]  # E4, B3, G3, E4, G4, E4
    timings = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    n = int(SR * DURATION)
    output = np.zeros(n, dtype=np.float32)

    for note, t in zip(notes, timings, strict=False):
        start = int(t * SR)
        pluck = karplus_strong(note, DURATION - t, SR, brightness=0.7)
        # Add harmonics (pickup resonance)
        idx = np.arange(len(pluck))
        t_arr = idx / SR
        pluck += (
            0.15 * np.sin(2 * np.pi * note * 3 * t_arr).astype(np.float32) * np.exp(-t_arr / 0.5).astype(np.float32)
        )
        end = min(start + len(pluck), n)
        output[start:end] += pluck[: end - start]

    # Soft clipping (tube amp simulation)
    output = np.tanh(output * 2.0) * 0.7

    # Add 60Hz hum (single-coil pickup)
    t = np.arange(n) / SR
    hum = 0.02 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
    output += hum

    return output / (np.max(np.abs(output)) + 1e-10) * 0.8


def generate_acoustic_guitar() -> np.ndarray:
    """Acoustic guitar — Karplus-Strong with body resonance."""
    notes = [220.0, 246.94, 261.63, 293.66, 329.63, 293.66, 261.63, 246.94]
    timings = [0.0, 0.35, 0.7, 1.05, 1.4, 1.75, 2.1, 2.45]

    n = int(SR * DURATION)
    output = np.zeros(n, dtype=np.float32)

    for note, t in zip(notes, timings, strict=False):
        start = int(t * SR)
        pluck = karplus_strong(note, DURATION - t, SR, brightness=0.4)
        end = min(start + len(pluck), n)
        output[start:end] += pluck[: end - start]

    # Body resonance filter (broad peaks at ~200Hz and ~400Hz)
    t = np.arange(n) / SR
    body_200 = np.convolve(
        output,
        np.exp(-np.arange(512) / 100.0).astype(np.float32)
        * np.sin(2 * np.pi * 200 * np.arange(512) / SR).astype(np.float32),
        mode="same",
    )
    body_400 = np.convolve(
        output,
        np.exp(-np.arange(256) / 50.0).astype(np.float32)
        * np.sin(2 * np.pi * 400 * np.arange(256) / SR).astype(np.float32),
        mode="same",
    )
    output = output + 0.3 * body_200.astype(np.float32) + 0.15 * body_400.astype(np.float32)

    return output / (np.max(np.abs(output)) + 1e-10) * 0.75


def generate_bass_guitar() -> np.ndarray:
    """Bass guitar — low-pitched Karplus-Strong with warm tone."""
    notes = [82.41, 98.0, 110.0, 82.41, 73.42, 82.41]
    timings = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    n = int(SR * DURATION)
    output = np.zeros(n, dtype=np.float32)

    for note, t in zip(notes, timings, strict=False):
        start = int(t * SR)
        pluck = karplus_strong(note, DURATION - t, SR, brightness=0.3)
        end = min(start + len(pluck), n)
        output[start:end] += pluck[: end - start]

    # Warm roll-off (finger tone — cut above 2kHz)
    from scipy.signal import butter, lfilter

    b, a = butter(4, 2000 / (SR / 2), btype="low")
    output = lfilter(b, a, output).astype(np.float32)

    return output / (np.max(np.abs(output)) + 1e-10) * 0.85


def generate_drum_kit() -> np.ndarray:
    """Drum kit — kick, snare, hi-hat pattern."""
    n = int(SR * DURATION)
    output = np.zeros(n, dtype=np.float32)
    np.arange(n) / SR

    # BPM = 120, 8th notes
    beat_interval = 0.5  # seconds

    for beat in range(int(DURATION / beat_interval)):
        pos = int(beat * beat_interval * SR)
        if pos >= n:
            break

        # Kick on 1 and 3
        if beat % 4 in (0, 4):
            kick_len = min(int(0.15 * SR), n - pos)
            kt = np.arange(kick_len) / SR
            freq_env = 150 + 200 * np.exp(-kt / 0.008)
            kick = np.sin(2 * np.pi * np.cumsum(freq_env / SR)) * np.exp(-kt / 0.1)
            kick += 0.3 * np.exp(-kt / 0.002) * np.random.randn(kick_len)  # click
            output[pos : pos + kick_len] += kick.astype(np.float32) * 0.8

        # Snare on 2 and 4
        if beat % 4 in (2, 6):
            snare_len = min(int(0.12 * SR), n - pos)
            st = np.arange(snare_len) / SR
            body = np.sin(2 * np.pi * 200 * st) * np.exp(-st / 0.04)
            snares = np.random.randn(snare_len) * np.exp(-st / 0.06) * 0.5
            snare = body + snares
            output[pos : pos + snare_len] += snare.astype(np.float32) * 0.7

        # Hi-hat on every 8th
        hh_len = min(int(0.05 * SR), n - pos)
        ht = np.arange(hh_len) / SR
        # Hi-hat = band-passed noise
        hh_noise = np.random.randn(hh_len)
        from scipy.signal import butter, lfilter

        b, a = butter(2, [6000 / (SR / 2), 12000 / (SR / 2)], btype="band")
        hh = lfilter(b, a, hh_noise).astype(np.float32) * np.exp(-ht / 0.02).astype(np.float32)
        output[pos : pos + hh_len] += hh * 0.3

    return output / (np.max(np.abs(output)) + 1e-10) * 0.8


def generate_piano() -> np.ndarray:
    """Piano — inharmonic partials with hammer action."""
    notes = [261.63, 329.63, 392.0, 523.25, 392.0, 329.63]
    timings = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    n = int(SR * DURATION)
    output = np.zeros(n, dtype=np.float32)

    for note, t_start in zip(notes, timings, strict=False):
        start = int(t_start * SR)
        remaining = DURATION - t_start
        note_len = int(remaining * SR)
        note_len = min(note_len, n - start)
        t = np.arange(note_len) / SR

        tone = np.zeros(note_len, dtype=np.float64)
        for h in range(1, 12):
            # Piano inharmonicity: f_n = f0 * n * sqrt(1 + B * n^2)
            B = 0.0004  # inharmonicity coefficient
            partial = note * h * np.sqrt(1 + B * h * h)
            amp = 1.0 / (h**0.6) * np.exp(-t * (0.5 + h * 0.3))
            tone += amp * np.sin(2 * np.pi * partial * t)

        # Hammer attack (brief broadband transient)
        hammer = np.exp(-t / 0.001) * 0.2 * np.random.randn(note_len)
        tone += hammer

        output[start : start + note_len] += tone.astype(np.float32)

    return output / (np.max(np.abs(output)) + 1e-10) * 0.75


def generate_male_vocal() -> np.ndarray:
    """Male vocal — glottal pulse with formant filtering."""
    n = int(SR * DURATION)
    t = np.arange(n) / SR

    # Fundamental with slight vibrato
    f0 = 130.0  # C3
    vibrato = 5.5 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum((f0 + vibrato) / SR)

    # Glottal pulse — more realistic than sine
    glottal = np.zeros(n, dtype=np.float64)
    for h in range(1, 20):
        amp = 1.0 / (h**1.0)  # -12dB/octave roll-off (typical voice)
        glottal += amp * np.sin(h * phase)

    # Formant filter ("ah" vowel: F1=700, F2=1200, F3=2600)
    formants = [(700, 130), (1200, 70), (2600, 160), (3300, 250)]
    output = np.zeros(n, dtype=np.float64)
    for f_center, bw in formants:
        from scipy.signal import butter, lfilter

        low = max(20, f_center - bw) / (SR / 2)
        high = min(0.99, (f_center + bw) / (SR / 2))
        if low >= high:
            continue
        b, a = butter(2, [low, high], btype="band")
        filtered = lfilter(b, a, glottal)
        output += filtered

    # Add breathiness
    breath = np.random.randn(n) * 0.03 * (1 + 0.5 * np.sin(2 * np.pi * 0.3 * t))
    output += breath

    # Amplitude envelope (gradual onset)
    env = np.ones(n)
    attack = int(0.05 * SR)
    env[:attack] = np.linspace(0, 1, attack)
    release = int(0.1 * SR)
    env[-release:] = np.linspace(1, 0, release)
    output *= env

    return (output / (np.max(np.abs(output)) + 1e-10) * 0.7).astype(np.float32)


def generate_female_vocal() -> np.ndarray:
    """Female vocal — higher pitch, different formants."""
    n = int(SR * DURATION)
    t = np.arange(n) / SR

    f0 = 260.0  # C4
    vibrato = 4.0 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum((f0 + vibrato) / SR)

    glottal = np.zeros(n, dtype=np.float64)
    for h in range(1, 15):
        amp = 1.0 / (h**1.2)
        glottal += amp * np.sin(h * phase)

    # Female "ee" vowel formants: F1=310, F2=2800, F3=3350
    formants = [(310, 45), (2800, 120), (3350, 200)]
    output = np.zeros(n, dtype=np.float64)
    for f_center, bw in formants:
        from scipy.signal import butter, lfilter

        low = max(20, f_center - bw) / (SR / 2)
        high = min(0.99, (f_center + bw) / (SR / 2))
        if low >= high:
            continue
        b, a = butter(2, [low, high], btype="band")
        filtered = lfilter(b, a, glottal)
        output += filtered

    breath = np.random.randn(n) * 0.04
    output += breath

    env = np.ones(n)
    attack = int(0.08 * SR)
    env[:attack] = np.linspace(0, 1, attack)
    release = int(0.15 * SR)
    env[-release:] = np.linspace(1, 0, release)
    output *= env

    return (output / (np.max(np.abs(output)) + 1e-10) * 0.7).astype(np.float32)


def generate_trumpet() -> np.ndarray:
    """Trumpet — brass synthesis with bright harmonics."""
    n = int(SR * DURATION)
    t = np.arange(n) / SR

    f0 = 466.16  # Bb4
    vibrato = 3.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum((f0 + vibrato) / SR)

    # Trumpet has strong odd and even harmonics
    output = np.zeros(n, dtype=np.float64)
    for h in range(1, 12):
        amp = 1.0 / (h**0.4)  # brass has lots of upper harmonics
        output += amp * np.sin(h * phase)

    # Brass attack — slower than string, with "blat"
    env = np.ones(n)
    attack = int(0.04 * SR)
    env[:attack] = np.linspace(0, 1, attack) ** 2

    # Lip buzzing noise at start
    buzz_len = int(0.03 * SR)
    buzz = np.random.randn(buzz_len) * 0.15 * np.exp(-np.arange(buzz_len) / (0.01 * SR))
    output[:buzz_len] += buzz

    output *= env
    return (output / (np.max(np.abs(output)) + 1e-10) * 0.75).astype(np.float32)


def generate_saxophone() -> np.ndarray:
    """Saxophone — reed instrument with warm harmonics."""
    n = int(SR * DURATION)
    t = np.arange(n) / SR

    f0 = 293.66  # D4
    vibrato = 5.0 * np.sin(2 * np.pi * 4.5 * t)
    phase = 2 * np.pi * np.cumsum((f0 + vibrato) / SR)

    # Sax has predominantly odd harmonics (closed pipe)
    output = np.zeros(n, dtype=np.float64)
    for h in range(1, 15):
        if h % 2 == 0:
            amp = 0.3 / (h**0.8)  # even harmonics weaker
        else:
            amp = 1.0 / (h**0.5)  # odd harmonics strong
        output += amp * np.sin(h * phase)

    # Reed buzz (noise modulation)
    reed_noise = np.random.randn(n) * 0.04
    output += reed_noise * np.abs(np.sin(phase))

    # Breath noise
    breath = np.random.randn(n) * 0.02
    output += breath

    env = np.ones(n)
    attack = int(0.06 * SR)
    env[:attack] = np.linspace(0, 1, attack)
    output *= env

    return (output / (np.max(np.abs(output)) + 1e-10) * 0.7).astype(np.float32)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generators = {
        "electric_guitar": ("Electric guitar with distortion and pickup", generate_electric_guitar),
        "acoustic_guitar": ("Acoustic guitar fingerpicking with body resonance", generate_acoustic_guitar),
        "bass_guitar": ("Bass guitar fingerstyle pattern", generate_bass_guitar),
        "drum_kit": ("Drum kit — kick/snare/hi-hat pattern at 120 BPM", generate_drum_kit),
        "piano": ("Piano chord progression with inharmonic partials", generate_piano),
        "male_vocal": ("Male vocal 'ah' with vibrato at C3", generate_male_vocal),
        "female_vocal": ("Female vocal 'ee' with vibrato at C4", generate_female_vocal),
        "trumpet": ("Trumpet sustained Bb4 with brass attack", generate_trumpet),
        "saxophone": ("Saxophone sustained D4 with reed buzz", generate_saxophone),
    }

    print(f"Generating {len(generators)} realistic instrument samples → {OUTPUT_DIR}/")
    for name, (desc, gen_fn) in generators.items():
        audio = gen_fn()
        path = OUTPUT_DIR / f"{name}.wav"
        sf.write(str(path), audio, SR)
        rms = float(np.sqrt(np.mean(audio**2)))
        print(f"  ✅ {name}.wav — {desc} (RMS: {rms:.4f}, {len(audio) / SR:.1f}s)")

    print(f"\n🎵 All {len(generators)} samples generated at {SR}Hz")


if __name__ == "__main__":
    main()
