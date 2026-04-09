"""
transient_analyzer.py — Transient Attack/Decay Analysis

Measures attack/decay characteristics to distinguish percussive vs sustained
instruments: attack time, decay time, transient sharpness, attack centroid.

Extracted from instrument_classifier.py for single-responsibility compliance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransientProfile:
    """Attack and decay characteristics of a sound event."""

    attack_time_ms: float  # time to peak (0→peak)
    decay_time_ms: float  # time from peak to -20dB
    is_percussive: bool  # True if attack < 10ms
    is_sustained: bool  # True if decay > 500ms
    transient_sharpness: float  # 0-1; 1 = extremely sharp like a pick
    attack_frequency_centroid_hz: float  # where the transient energy is concentrated


class TransientAnalyzer:
    """Measures attack/decay characteristics to distinguish percussive vs sustained."""

    def __init__(self, sample_rate: int = 48_000):
        self._sr = sample_rate

    def analyze(self, audio: np.ndarray) -> TransientProfile:
        envelope = self._compute_envelope(audio)
        if len(envelope) == 0 or np.max(envelope) < 1e-8:
            return TransientProfile(
                attack_time_ms=0,
                decay_time_ms=0,
                is_percussive=False,
                is_sustained=False,
                transient_sharpness=0,
                attack_frequency_centroid_hz=0,
            )

        peak_idx = int(np.argmax(envelope))
        peak_val = envelope[peak_idx]

        # Attack: time from 10% to 90% of peak
        threshold_10 = peak_val * 0.1
        threshold_90 = peak_val * 0.9
        attack_start = 0
        for i in range(peak_idx):
            if envelope[i] >= threshold_10:
                attack_start = i
                break
        attack_end = peak_idx
        for i in range(peak_idx):
            if envelope[i] >= threshold_90:
                attack_end = i
                break
        attack_samples = max(1, attack_end - attack_start)
        attack_ms = (attack_samples / self._sr) * 1000

        # Decay: time from peak to -20dB
        decay_threshold = peak_val * 0.1  # -20dB
        decay_end = len(envelope) - 1
        for i in range(peak_idx, len(envelope)):
            if envelope[i] <= decay_threshold:
                decay_end = i
                break
        decay_ms = ((decay_end - peak_idx) / self._sr) * 1000

        # Attack frequency centroid (spectral content of the first 10ms)
        attack_window = audio[max(0, peak_idx - int(0.005 * self._sr)) : peak_idx + int(0.005 * self._sr)]
        if len(attack_window) > 64:
            spectrum = np.abs(np.fft.rfft(attack_window * np.hanning(len(attack_window))))
            freqs = np.fft.rfftfreq(len(attack_window), d=1.0 / self._sr)
            power = spectrum**2 + 1e-10
            attack_centroid = float(np.sum(freqs * power) / np.sum(power))
        else:
            attack_centroid = 0.0

        # Transient sharpness: ratio of peak value to attack time
        sharpness = min(1.0, 10.0 / (attack_ms + 0.1))

        return TransientProfile(
            attack_time_ms=attack_ms,
            decay_time_ms=decay_ms,
            is_percussive=attack_ms < 10.0,
            is_sustained=decay_ms > 500.0,
            transient_sharpness=sharpness,
            attack_frequency_centroid_hz=attack_centroid,
        )

    def _compute_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Amplitude envelope via rectification + smoothing."""
        rectified = np.abs(audio)
        # Smooth with 1ms window
        window_size = max(1, int(self._sr * 0.001))
        if len(rectified) < window_size:
            return rectified
        kernel = np.ones(window_size) / window_size
        return np.convolve(rectified, kernel, mode="same")
