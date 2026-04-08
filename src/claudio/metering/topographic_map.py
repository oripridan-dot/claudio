"""
topographic_map.py — Topographic Frequency Map Engine

3D frequency collision analysis for masking/mud detection between sources.
Extracted from semantic_metering.py for 300-line compliance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FreqCollisionZone:
    """A detected frequency range where two sources are masking each other."""
    freq_hz_low:  float
    freq_hz_high: float
    source_a:     str
    source_b:     str
    collision_db: float   # energy overlap in dB — higher = more mud
    severity:     str     # "low", "medium", "critical"


@dataclass
class TopographicFreqMapFrame:
    """
    3D topographic map data for the frequency collision display.

    UI rendering:
      - X axis: frequency (20 Hz – 20 kHz, log scale)
      - Y axis: signal energy (dB)
      - Z/colour: collision severity (blue=clear, yellow=warning, red=critical)
      - Collision zones are highlighted with a glowing red mesh
    """
    source_spectra:    dict[str, np.ndarray]   # source_id → 1024-bin magnitude spectrum
    collision_zones:   list[FreqCollisionZone]
    total_mud_score:   float  # 0.0 (crystal clear) – 1.0 (heavily masked)
    freq_bins_hz:      np.ndarray              # 1024-element log-frequency axis


class TopographicFreqMap:
    """
    Analyses frequency spectra from all active sources and identifies
    masking collisions — the "mud zones" in the mix.

    Physics:
      - Two sources mask each other when their simultaneous energy in a
        critical band (Bark scale) exceeds the threshold of masking.
      - Simplified model: collision if overlap energy > 6 dB in any 1/3-oct band.
    """

    FFT_SIZE    = 2048
    N_BINS      = 1024
    COLLISION_DB = 6.0   # overlap threshold for "muddy" masking

    def __init__(self, sample_rate: int = 192_000) -> None:
        self._sr = sample_rate
        self._freq_bins = np.geomspace(20, sample_rate / 2, self.N_BINS)

    def compute(
        self,
        source_buffers: dict[str, np.ndarray],  # source_id → mono audio block
    ) -> TopographicFreqMapFrame:
        if not source_buffers:
            return TopographicFreqMapFrame(
                source_spectra={}, collision_zones=[],
                total_mud_score=0.0, freq_bins_hz=self._freq_bins,
            )

        source_spectra: dict[str, np.ndarray] = {}
        for sid, buf in source_buffers.items():
            if len(buf) < self.FFT_SIZE:
                buf = np.pad(buf, (0, self.FFT_SIZE - len(buf)))
            window  = np.hanning(self.FFT_SIZE)
            spectrum = np.abs(np.fft.rfft(buf[:self.FFT_SIZE] * window))
            # Interpolate to N_BINS log-frequency bins
            fft_freqs  = np.fft.rfftfreq(self.FFT_SIZE, d=1.0 / self._sr)
            log_mag    = np.interp(self._freq_bins, fft_freqs, spectrum)
            source_spectra[sid] = 20 * np.log10(log_mag + 1e-10)

        # Detect collision zones
        collision_zones = []
        ids = list(source_spectra.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sid_a = ids[i]
                sid_b = ids[j]
                s_a   = source_spectra[sid_a]
                s_b   = source_spectra[sid_b]
                overlap = np.minimum(s_a, s_b)
                # Collision: both sources > −40 dBFS and overlap > threshold
                active  = (s_a > -40) & (s_b > -40)
                collide = active & (overlap > self.COLLISION_DB)
                if np.any(collide):
                    col_freqs = self._freq_bins[collide]
                    col_db    = float(np.mean(overlap[collide]))
                    severity  = (
                        "critical" if col_db > 18
                        else "medium" if col_db > 10
                        else "low"
                    )
                    collision_zones.append(FreqCollisionZone(
                        freq_hz_low  = float(col_freqs.min()),
                        freq_hz_high = float(col_freqs.max()),
                        source_a     = sid_a,
                        source_b     = sid_b,
                        collision_db = col_db,
                        severity     = severity,
                    ))

        mud_score = min(1.0, len(collision_zones) * 0.15 +
                        sum(z.collision_db / 40.0 for z in collision_zones))

        return TopographicFreqMapFrame(
            source_spectra  = source_spectra,
            collision_zones = collision_zones,
            total_mud_score = float(mud_score),
            freq_bins_hz    = self._freq_bins,
        )
