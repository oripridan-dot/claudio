"""
spectral_extractor.py — Acoustic Spectral Feature Extraction

Extracts compact acoustic fingerprints from short audio windows:
mel-spectrogram, MFCC, spectral centroid, rolloff, flatness, bandwidth,
zero-crossing rate, RMS energy, and harmonic ratio.

Extracted from instrument_classifier.py for single-responsibility compliance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralFingerprint:
    """Compact acoustic signature extracted from a short audio window."""
    mfcc_coefficients: np.ndarray          # (13,) mel-frequency cepstral coefficients
    spectral_centroid_hz: float            # brightness indicator
    spectral_rolloff_hz: float             # frequency below which 85% energy lives
    spectral_flatness: float               # 0=tonal, 1=noise-like
    spectral_bandwidth_hz: float           # spread of the spectrum
    zero_crossing_rate: float              # transient/noise indicator
    rms_energy: float                      # loudness
    harmonic_ratio: float                  # harmonic vs noise energy (0-1)


class SpectralExtractor:
    """Extracts acoustic fingerprint from a short audio window."""

    def __init__(self, sample_rate: int = 48_000, n_fft: int = 2048, n_mels: int = 128):
        self._sr = sample_rate
        self._n_fft = n_fft
        self._n_mels = n_mels
        # Pre-compute mel filterbank
        self._mel_fb = self._build_mel_filterbank(n_mels, n_fft, sample_rate)

    def extract(self, audio: np.ndarray) -> SpectralFingerprint:
        if len(audio) < self._n_fft:
            audio = np.pad(audio, (0, self._n_fft - len(audio)))

        window = np.hanning(self._n_fft)
        windowed = audio[:self._n_fft] * window
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(self._n_fft, d=1.0 / self._sr)
        power = spectrum ** 2 + 1e-10

        # Spectral centroid
        centroid = float(np.sum(freqs * power) / np.sum(power))

        # Spectral rolloff (85% energy threshold)
        cumulative = np.cumsum(power)
        rolloff_idx = np.searchsorted(cumulative, 0.85 * cumulative[-1])
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # Spectral flatness (geometric mean / arithmetic mean)
        log_power = np.log(power + 1e-10)
        geo_mean = np.exp(np.mean(log_power))
        arith_mean = np.mean(power)
        flatness = float(geo_mean / (arith_mean + 1e-10))

        # Spectral bandwidth
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / np.sum(power)))

        # Zero crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio[:self._n_fft])))) / 2)

        # RMS energy
        rms = float(np.sqrt(np.mean(audio[:self._n_fft] ** 2)))

        # MFCC (simplified — 13 coefficients)
        mel_spectrum = self._mel_fb @ power[:self._mel_fb.shape[1]]
        log_mel = np.log(mel_spectrum + 1e-10)
        mfcc = self._dct(log_mel, 13)

        # Harmonic ratio (autocorrelation-based)
        ac = np.correlate(windowed, windowed, mode='full')
        ac = ac[len(ac) // 2:]
        if ac[0] > 0:
            harmonic_ratio = float(np.max(ac[20:]) / ac[0])
        else:
            harmonic_ratio = 0.0

        return SpectralFingerprint(
            mfcc_coefficients=mfcc,
            spectral_centroid_hz=centroid,
            spectral_rolloff_hz=rolloff,
            spectral_flatness=flatness,
            spectral_bandwidth_hz=bandwidth,
            zero_crossing_rate=zcr,
            rms_energy=rms,
            harmonic_ratio=harmonic_ratio,
        )

    def _build_mel_filterbank(self, n_mels: int, n_fft: int, sr: int) -> np.ndarray:
        """Build a mel-scale triangular filterbank matrix."""
        n_bins = n_fft // 2 + 1
        f_min, f_max = 20.0, sr / 2.0
        mel_min = 2595 * math.log10(1 + f_min / 700)
        mel_max = 2595 * math.log10(1 + f_max / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        fb = np.zeros((n_mels, n_bins))
        for m in range(1, n_mels + 1):
            f_left = bin_points[m - 1]
            f_center = bin_points[m]
            f_right = bin_points[m + 1]
            for k in range(f_left, f_center):
                if f_center > f_left:
                    fb[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right > f_center:
                    fb[m - 1, k] = (f_right - k) / (f_right - f_center)
        return fb

    @staticmethod
    def _dct(x: np.ndarray, n_out: int) -> np.ndarray:
        """Type-II DCT (simplified)."""
        N = len(x)
        result = np.zeros(n_out)
        for k in range(n_out):
            result[k] = np.sum(x * np.cos(np.pi * k * (2 * np.arange(N) + 1) / (2 * N)))
        return result
