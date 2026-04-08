"""
signal_flow_metrics.py — Audio Quality Measurement Utilities

Provides signal-level quality measurements: SNR, THD+N, phase coherence.
Used by the signal flow simulator to evaluate pipeline output quality.
"""
from __future__ import annotations

import math

import numpy as np


def measure_snr(signal: np.ndarray, reference: np.ndarray) -> float:
    """Signal-to-noise ratio in dB (signal vs scaled reference)."""
    min_len = min(len(signal), len(reference))
    if min_len == 0:
        return 0.0
    sig = signal[:min_len].astype(np.float64)
    ref = reference[:min_len].astype(np.float64)
    scale = np.dot(sig, ref) / (np.dot(ref, ref) + 1e-30)
    noise = sig - scale * ref
    sig_power = np.mean(sig ** 2) + 1e-30
    noise_power = np.mean(noise ** 2) + 1e-30
    return 10 * math.log10(sig_power / noise_power)


def measure_thdn(signal: np.ndarray) -> float:
    """Total harmonic distortion + noise as a percentage."""
    if len(signal) < 256:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal))
    if np.max(spectrum) < 1e-10:
        return 0.0
    fund_idx = int(np.argmax(spectrum[1:])) + 1
    fund_power = spectrum[fund_idx] ** 2
    total_power = np.sum(spectrum ** 2)
    noise_power = total_power - fund_power
    return math.sqrt(max(0, noise_power) / (fund_power + 1e-30)) * 100.0


def measure_phase_coherence(left: np.ndarray, right: np.ndarray) -> float:
    """Stereo phase correlation coefficient (0..1)."""
    if len(left) < 64 or len(right) < 64:
        return 0.0
    min_len = min(len(left), len(right))
    left_c = left[:min_len].astype(np.float64) - np.mean(left[:min_len])
    right_c = right[:min_len].astype(np.float64) - np.mean(right[:min_len])
    num = np.sum(left_c * right_c)
    den = math.sqrt(np.sum(left_c ** 2) * np.sum(right_c ** 2)) + 1e-30
    return abs(num / den)
