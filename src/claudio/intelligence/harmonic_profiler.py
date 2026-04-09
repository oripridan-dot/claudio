"""
harmonic_profiler.py — Partial Series Harmonic Analysis

Analyses the partial series of pitched instruments to identify timbre:
fundamental estimation (YIN-lite), partial amplitudes, inharmonicity
coefficient, and odd/even partial ratio.

Extracted from instrument_classifier.py for single-responsibility compliance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HarmonicProfile:
    """Partial series analysis — identifies instrument via overtone structure."""

    fundamental_hz: float
    n_partials: int
    partial_amplitudes: np.ndarray  # relative amplitude of each partial
    inharmonicity_coefficient: float  # 0=perfect harmonic, >0=stiff string (piano/bell)
    odd_even_ratio: float  # >1 = odd-dominant (clarinet/square), <1 = even-dominant


class HarmonicProfiler:
    """Analyses the partial series of a pitched instrument to identify timbre."""

    def __init__(self, sample_rate: int = 48_000, n_fft: int = 4096):
        self._sr = sample_rate
        self._n_fft = n_fft

    def analyze(self, audio: np.ndarray, fundamental_hint: float = 0.0) -> HarmonicProfile:
        if len(audio) < self._n_fft:
            audio = np.pad(audio, (0, self._n_fft - len(audio)))

        window = np.hanning(self._n_fft)
        spectrum = np.abs(np.fft.rfft(audio[: self._n_fft] * window))
        freqs = np.fft.rfftfreq(self._n_fft, d=1.0 / self._sr)

        # Find fundamental via autocorrelation (YIN-lite)
        if fundamental_hint <= 0:
            f0 = self._estimate_f0(audio[: self._n_fft])
        else:
            f0 = fundamental_hint

        if f0 < 20:
            return HarmonicProfile(
                fundamental_hz=0,
                n_partials=0,
                partial_amplitudes=np.array([]),
                inharmonicity_coefficient=0,
                odd_even_ratio=1.0,
            )

        # Extract partial amplitudes
        max_partials = min(20, int((self._sr / 2) / f0))
        amplitudes = np.zeros(max_partials)
        actual_freqs = np.zeros(max_partials)

        for n in range(1, max_partials + 1):
            expected_freq = f0 * n
            # Search in ±3% window around expected frequency
            tolerance = expected_freq * 0.03
            mask = (freqs >= expected_freq - tolerance) & (freqs <= expected_freq + tolerance)
            if np.any(mask):
                peak_idx = np.argmax(spectrum[mask])
                amplitudes[n - 1] = float(spectrum[mask][peak_idx])
                actual_freqs[n - 1] = float(freqs[mask][peak_idx])

        # Normalize amplitudes
        if amplitudes[0] > 0:
            amplitudes = amplitudes / amplitudes[0]

        # Inharmonicity: how much do partials deviate from integer multiples?
        inharmonicity = 0.0
        count = 0
        for n in range(2, max_partials + 1):
            if actual_freqs[n - 1] > 0 and f0 > 0:
                expected = f0 * n
                deviation = abs(actual_freqs[n - 1] - expected) / expected
                inharmonicity += deviation
                count += 1
        if count > 0:
            inharmonicity /= count

        # Odd/even ratio
        odd_energy = float(np.sum(amplitudes[0::2] ** 2))
        even_energy = float(np.sum(amplitudes[1::2] ** 2)) + 1e-10
        odd_even = odd_energy / even_energy

        return HarmonicProfile(
            fundamental_hz=f0,
            n_partials=int(np.sum(amplitudes > 0.01)),
            partial_amplitudes=amplitudes,
            inharmonicity_coefficient=inharmonicity,
            odd_even_ratio=odd_even,
        )

    def _estimate_f0(self, audio: np.ndarray) -> float:
        """YIN-lite pitch estimation."""
        max_lag = int(self._sr / 60)  # 60 Hz minimum
        min_lag = int(self._sr / 4000)  # 4kHz maximum
        if max_lag >= len(audio) // 2:
            max_lag = len(audio) // 2 - 1

        # Difference function
        d = np.zeros(max_lag)
        for tau in range(min_lag, max_lag):
            diff = audio[: len(audio) - tau] - audio[tau : 2 * (len(audio) - tau) // 2 + tau]
            if len(diff) > 0:
                d[tau] = np.sum(diff[: len(audio) - max_lag] ** 2)

        # Cumulative mean normalized difference
        d_prime = np.ones(max_lag)
        running_sum = 0.0
        for tau in range(min_lag, max_lag):
            running_sum += d[tau]
            if running_sum > 0:
                d_prime[tau] = d[tau] * tau / running_sum

        # Find first dip below threshold
        threshold = 0.15
        for tau in range(min_lag, max_lag - 1):
            if d_prime[tau] < threshold and d_prime[tau] < d_prime[tau + 1]:
                return self._sr / tau

        # Fallback: absolute minimum
        if max_lag > min_lag:
            best_tau = min_lag + int(np.argmin(d_prime[min_lag:max_lag]))
            if best_tau > 0:
                return self._sr / best_tau
        return 0.0
