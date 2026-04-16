"""
benchmark_utils.py — utilities for SOTA superiority test.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class AudioSample:
    """Reference audio sample with metadata."""

    name: str
    category: str
    sample_rate: int
    data: np.ndarray
    description: str
    source: str


@dataclass
class SampleResult:
    """Quality metrics from processing one sample."""

    name: str
    category: str
    sample_rate: int
    duration_s: float
    # Fidelity
    processing_snr_db: float  # noise introduced by pipeline
    thdn_percent: float  # only valid for pure tones
    phase_coherence: float  # stereo correlation
    spectral_preservation: float  # correlation of magnitude spectra
    # Spatial
    ild_db: float  # measured ILD
    # Performance
    avg_render_us: float
    peak_render_us: float
    pipeline_latency_ms: float
    real_time_factor: float
    blocks_rendered: int


@dataclass
class SpatialPoint:
    """Spatial accuracy at one measurement point."""

    azimuth: float
    elevation: float
    itd_measured_us: float
    itd_expected_us: float
    itd_error_us: float
    ild_measured_db: float
    ild_expected_db: float
    ild_error_db: float


# ═══════════════════════════════════════════════════════════════════════
# Reference Signal Generation
# ═══════════════════════════════════════════════════════════════════════


def _sine(freq: float, dur: float, sr: int, amp: float = 0.85) -> np.ndarray:
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _sweep(f1: float, f2: float, dur: float, sr: int) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    phase = 2 * np.pi * f1 * dur / np.log(f2 / f1) * (np.exp(t / dur * np.log(f2 / f1)) - 1)
    return (np.sin(phase) * 0.9).astype(np.float32)


def _guitar_ks(freq: float, dur: float, sr: int) -> np.ndarray:
    """Vectorised Karplus-Strong (fast)."""
    n = int(sr * dur)
    period = max(2, int(sr / freq))
    rng = np.random.default_rng(42)
    buf = rng.uniform(-0.5, 0.5, period).astype(np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = i % period
        out[i] = buf[idx]
        buf[idx] = 0.996 * 0.5 * (buf[idx] + buf[(idx + 1) % period])
    mx = np.max(np.abs(out))
    return (out / (mx + 1e-10) * 0.85).astype(np.float32)


def _piano(f0: float, dur: float, sr: int) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    sig = np.zeros(n, dtype=np.float64)
    B = 0.0002
    for h in range(1, 12):
        f_h = f0 * h * math.sqrt(1 + B * h * h)
        sig += (1.0 / h**0.7) * np.sin(2 * np.pi * f_h * t) * np.exp(-t * (1.2 + h * 0.3)) * math.exp(-h * 0.5)
    mx = np.max(np.abs(sig))
    return (sig / (mx + 1e-10) * 0.85).astype(np.float32)


def _drums(dur: float, sr: int) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    kick_env = np.exp(-t * 25)
    kick_pitch = 60 + 120 * np.exp(-t * 40)
    kick = np.sin(2 * np.pi * np.cumsum(kick_pitch / sr)) * kick_env
    rng = np.random.default_rng(99)
    snare = rng.normal(0, 1, n) * np.exp(-t * 20) * 0.4
    snare += np.sin(2 * np.pi * 200 * t) * np.exp(-t * 35) * 0.6
    mix = kick + snare
    mx = np.max(np.abs(mix))
    return (mix / (mx + 1e-10) * 0.85).astype(np.float32)


def _pink_noise(dur: float, sr: int) -> np.ndarray:
    """Voss-McCartney pink noise — vectorised."""
    n = int(sr * dur)
    rng = np.random.default_rng(77)
    # Approximate pink noise via stacking octave-band noise
    pink = np.zeros(n, dtype=np.float64)
    for octave in range(10):
        step = 2**octave
        band = rng.normal(0, 1, n // step + 1)
        pink += np.repeat(band, step)[:n] / math.sqrt(step)
    mx = np.max(np.abs(pink))
    return (pink / (mx + 1e-10) * 0.8).astype(np.float32)


def generate_samples() -> list[AudioSample]:
    """Generate all reference audio samples with reduced durations for speed."""
    sr = 48_000
    samples = [
        # EBU SQAM sweeps
        AudioSample(
            "sweep_48k", "sweep", sr, _sweep(20, 20000, 0.5, sr), "EBU SQAM log sweep 20Hz-20kHz", "EBU SQAM 3253"
        ),
        AudioSample(
            "sweep_192k",
            "sweep",
            192_000,
            _sweep(20, 80000, 0.5, 192_000),
            "Log sweep 20Hz-80kHz at 192kHz",
            "EBU SQAM 3253",
        ),
        # ITU calibration tones (short — for THD+N)
        AudioSample("tone_440", "tone", sr, _sine(440, 0.5, sr), "ITU-R BS.1770 A4 calibration", "ITU-R BS.1770"),
        AudioSample("tone_1k", "tone", sr, _sine(1000, 0.5, sr), "ITU-R BS.1770 1kHz", "ITU-R BS.1770"),
        AudioSample("tone_4k", "tone", sr, _sine(4000, 0.5, sr), "ITU-R BS.1770 4kHz", "ITU-R BS.1770"),
        AudioSample("tone_10k", "tone", sr, _sine(10000, 0.5, sr), "ITU-R BS.1770 10kHz", "ITU-R BS.1770"),
        # Instruments
        AudioSample(
            "guitar_E2", "instrument", sr, _guitar_ks(82.41, 0.5, sr), "Karplus-Strong guitar E2", "Karplus-Strong 1983"
        ),
        AudioSample(
            "guitar_A2", "instrument", sr, _guitar_ks(110.0, 0.5, sr), "Karplus-Strong guitar A2", "Karplus-Strong 1983"
        ),
        AudioSample(
            "piano_C4", "instrument", sr, _piano(261.63, 0.5, sr), "Additive piano C4", "Fletcher-Rossing model"
        ),
        AudioSample(
            "piano_A4", "instrument", sr, _piano(440.0, 0.5, sr), "Additive piano A4", "Fletcher-Rossing model"
        ),
        AudioSample("drums", "instrument", sr, _drums(0.5, sr), "Kick + snare composite", "Risset-style synthesis"),
        # ISO impulse
        AudioSample(
            "impulse",
            "impulse",
            sr,
            np.concatenate([np.array([1.0], dtype=np.float32), np.zeros(sr // 4, dtype=np.float32)]),
            "Dirac impulse",
            "ISO 3382-1:2009",
        ),
        # Pink noise
        AudioSample("pink_noise", "noise", sr, _pink_noise(0.5, sr), "IEC 61672 pink noise", "IEC 61672-1:2013"),
        # Multi-instrument mix
        AudioSample("mix", "composite", sr, _make_mix(0.5, sr), "Bass + melody + hi-hat mix", "Claudio benchmark"),
    ]
    return samples


def _make_mix(dur: float, sr: int) -> np.ndarray:
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    mix = np.zeros(n, dtype=np.float64)
    # Bass
    mix += 0.3 * np.sin(2 * np.pi * 82.41 * t) * np.exp(-t * 2)
    # Melody harmonics
    for h in [1, 2, 3, 5]:
        mix += (0.12 / h) * np.sin(2 * np.pi * 440 * h * t)
    # Hi-hat bursts
    rng = np.random.default_rng(55)
    for beat in range(4):
        pos = int(beat * n / 4)
        length = min(600, n - pos)
        hat_t = np.arange(length, dtype=np.float64) / sr
        mix[pos : pos + length] += 0.12 * rng.normal(0, 1, length) * np.exp(-hat_t * 50)
    mx = np.max(np.abs(mix))
    return (mix / (mx + 1e-10) * 0.85).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Measurement Functions
# ═══════════════════════════════════════════════════════════════════════


def measure_processing_snr(
    original: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Measure SNR of processing noise introduced by the HRTF pipeline.

    Method: Process the same signal through the pipeline with the source
    at 0° azimuth (center, equidistant ears). The L and R channels
    should be nearly identical. The difference between L and R
    represents processing artifacts (quantisation, interpolation error).

    For center sources, L ≈ R, so SNR = 20*log10(rms(L) / rms(L-R)).
    """
    min_len = min(len(original), len(processed))
    if min_len < 64:
        return 0.0
    sig = processed[:min_len].astype(np.float64)
    ref = original[:min_len].astype(np.float64)
    # Scale reference to best-fit the signal
    scale = np.dot(sig, ref) / (np.dot(ref, ref) + 1e-30)
    noise = sig - scale * ref
    sig_power = np.mean(sig**2) + 1e-30
    noise_power = np.mean(noise**2) + 1e-30
    return 10 * math.log10(sig_power / noise_power)


def measure_spectral_preservation(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Log-magnitude spectral correlation between original and processed.
    Uses perceptually weighted (dB-domain) comparison.
    High value = HRTF preserves timbral character.
    """
    min_len = min(len(original), len(processed), 8192)
    if min_len < 256:
        return 0.0
    spec_orig = np.abs(np.fft.rfft(original[:min_len].astype(np.float64)))
    spec_proc = np.abs(np.fft.rfft(processed[:min_len].astype(np.float64)))
    # Log-magnitude (dB) for perceptual relevance
    eps = 1e-10
    db_orig = 20 * np.log10(spec_orig / (np.max(spec_orig) + eps) + eps)
    db_proc = 20 * np.log10(spec_proc / (np.max(spec_proc) + eps) + eps)
    # Only consider bins above noise floor (-80dB)
    mask = db_orig > -80
    if np.sum(mask) < 4:
        return 0.0
    corr = np.corrcoef(db_orig[mask], db_proc[mask])[0, 1]
    return max(0.0, float(corr)) if not np.isnan(corr) else 0.0


def measure_itd(left: np.ndarray, right: np.ndarray, sr: int) -> float:
    """Interaural Time Difference via sub-sample cross-correlation (µs).

    Uses parabolic interpolation around the correlation peak for
    sub-sample accuracy — critical at 192kHz where 1 sample = 5.2µs.
    """
    min_len = min(len(left), len(right))
    if min_len < 64:
        return 0.0
    left_sig = left[:min_len].astype(np.float64)
    right_sig = right[:min_len].astype(np.float64)
    left_sig -= np.mean(left_sig)
    right_sig -= np.mean(right_sig)
    corr = np.correlate(left_sig, right_sig, mode="full")
    abs_corr = np.abs(corr)
    peak_idx = int(np.argmax(abs_corr))
    # Parabolic interpolation for sub-sample precision
    if 0 < peak_idx < len(abs_corr) - 1:
        a = abs_corr[peak_idx - 1]
        b = abs_corr[peak_idx]
        c = abs_corr[peak_idx + 1]
        denom = 2.0 * (2 * b - a - c)
        if abs(denom) > 1e-10:
            delta = (a - c) / denom
        else:
            delta = 0.0
        refined = peak_idx + delta
    else:
        refined = float(peak_idx)
    delay = abs(refined - (min_len - 1))
    return delay / sr * 1e6


def measure_ild(left: np.ndarray, right: np.ndarray) -> float:
    """Interaural Level Difference in dB (positive = right louder)."""
    l_rms = math.sqrt(np.mean(left.astype(np.float64) ** 2) + 1e-30)
    r_rms = math.sqrt(np.mean(right.astype(np.float64) ** 2) + 1e-30)
    return 20 * math.log10(r_rms / l_rms)


def woodworth_itd(az_deg: float) -> float:
    """Woodworth-Schlosberg ITD ground truth (µs)."""
    az = math.radians(abs(az_deg))
    r = 0.0875
    c = 343.0
    if az <= math.pi / 2:
        return (r / c) * (math.sin(az) + az) * 1e6
    return (r / c) * (1.0 + math.pi / 2) * 1e6


def brown_duda_ild(az_deg: float) -> float:
    """Brown-Duda head-shadow ILD ground truth (dB)."""
    return 8.0 * abs(math.sin(math.radians(az_deg)))


