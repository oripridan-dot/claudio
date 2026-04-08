"""
acoustic_advisor.py — Acoustic Environment Advisor

Analyses a mono mic recording and detects acoustic pathologies:
  - Flutter echo: comb-filter peaks with regular spacing in the spectrum
  - Bass buildup: excessive energy below 200 Hz (room mode)
  - Comb filtering: mic too close to a reflective surface

All analysis runs on a short analysis window (≈1 second of audio).

Extracted from semantic_metering.py for single-responsibility compliance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AcousticAdvice:
    """A room-treatment or placement suggestion."""
    category:   str   # "flutter_echo", "bass_buildup", "comb_filter", "reflection"
    description: str
    action:     str   # concrete corrective action


class AcousticEnvironmentAdvisor:
    """
    Analyses a mono mic recording and detects acoustic pathologies:
      - Flutter echo: comb-filter peaks with regular spacing in the spectrum
      - Bass buildup: excessive energy below 200 Hz (room mode)
      - Comb filtering: mic too close to a reflective surface

    All analysis runs on a short analysis window (≈1 second of audio).
    """

    def analyse(
        self,
        audio_mono: np.ndarray,
        sample_rate: int = 48_000,
    ) -> list[AcousticAdvice]:
        advice: list[AcousticAdvice] = []

        if len(audio_mono) < sample_rate // 4:
            return advice

        window  = audio_mono[: sample_rate]  # first second
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        freqs    = np.fft.rfftfreq(len(window), d=1.0 / sample_rate)

        # Bass buildup: energy ratio below 200 Hz vs 200–2000 Hz
        bass_mask = freqs < 200
        mid_mask  = (freqs >= 200) & (freqs < 2000)
        if mid_mask.any() and bass_mask.any():
            bass_energy = float(np.mean(spectrum[bass_mask])) + 1e-10
            mid_energy  = float(np.mean(spectrum[mid_mask]))  + 1e-10
            if bass_energy / mid_energy > 3.0:
                advice.append(AcousticAdvice(
                    category="bass_buildup",
                    description=(
                        "Significant low-frequency room mode detected below 200 Hz. "
                        "Your room is reinforcing bass frequencies, causing mud."
                    ),
                    action=(
                        "Move your microphone stand away from room corners by at least 1 m. "
                        "Corner placement maximises room mode excitation. "
                        "Bass traps in the corner behind you will also help dramatically."
                    ),
                ))

        # Flutter echo: look for periodic spectral peaks (comb pattern)
        peak_intervals = self._detect_comb_pattern(spectrum, freqs)
        if peak_intervals and len(peak_intervals) >= 3:
            advice.append(AcousticAdvice(
                category="flutter_echo",
                description=(
                    "Flutter echo pattern detected — likely caused by two parallel "
                    "reflective surfaces (e.g., opposite walls or ceiling/floor)."
                ),
                action=(
                    "Rotate your microphone 45 degrees away from the glass window or "
                    "parallel wall. Hanging a duvet or acoustic panel on the reflective "
                    "surface opposite your mic will neutralise the flutter immediately."
                ),
            ))

        return advice

    @staticmethod
    def _detect_comb_pattern(
        spectrum: np.ndarray, freqs: np.ndarray, n_peaks: int = 8
    ) -> list[float]:
        """Detect regularly-spaced spectral peaks (comb filter signature)."""
        # Find top N peaks
        peak_indices = []
        threshold = np.mean(spectrum) + np.std(spectrum)
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > threshold and spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                peak_indices.append(i)
        if len(peak_indices) < 3:
            return []

        # Check for regular spacing
        peak_freqs = freqs[peak_indices[:n_peaks]]
        if len(peak_freqs) < 3:
            return []
        intervals  = np.diff(peak_freqs)
        if np.std(intervals) / (np.mean(intervals) + 1e-6) < 0.15:
            # Highly regular spacing → comb filter
            return intervals.tolist()
        return []
