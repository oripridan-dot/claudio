"""
intent_encoder.py — Semantic Intent Extraction from Raw Audio

Extracts the following per-frame intents from a raw audio stream:
  - F0 (fundamental frequency) via YIN pitch tracking
  - Spectral envelope via MFCC coefficients (compact timbre)
  - Loudness via A-weighted RMS energy
  - Onset/transient detection via spectral flux
  - Articulation features: vibrato depth/rate

Frame rate: 250 Hz (4ms frames) for real-time responsiveness.
Output: IntentFrame dataclass (~80 bytes per frame vs ~352 bytes raw PCM)

Bandwidth comparison at 44.1kHz mono 16-bit:
  Raw PCM:    44100 × 2 = 88,200 bytes/sec (~86 KB/s)
  Intent:     250 × ~80 = 20,000 bytes/sec  (~20 KB/s)
  With delta compression: ~2-5 KB/s typical
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class IntentFrame:
    """Single frame of semantic intent extracted from audio."""

    timestamp_ms: float
    f0_hz: float               # Fundamental frequency (0 = unvoiced)
    f0_confidence: float       # Pitch detection confidence [0, 1]
    loudness_db: float         # A-weighted RMS loudness in dB
    loudness_norm: float       # Normalised loudness [0, 1]
    spectral_centroid_hz: float
    mfcc: list[float] = field(default_factory=list)  # 13 MFCC coefficients (timbre)
    is_onset: bool = False     # Transient onset detected
    onset_strength: float = 0.0
    vibrato_rate_hz: float = 0.0
    vibrato_depth_cents: float = 0.0
    rms_energy: float = 0.0


FRAME_RATE_HZ = 250
F0_MIN_HZ = 32.7   # C1
F0_MAX_HZ = 4186.0  # C8
N_MFCC = 13


class IntentEncoder:
    """Extracts semantic intents from raw audio at 250 Hz frame rate.

    NOT thread-safe: maintains onset detection and vibrato history
    state between calls. Create one instance per audio stream.
    """

    def __init__(self, sample_rate: int = 44_100) -> None:
        self.sample_rate = sample_rate
        self.hop = sample_rate // FRAME_RATE_HZ
        # 8× hop for YIN window — needed to detect F0 down to ~62Hz (B1)
        # At 44.1kHz: frame_len=1408, half=704, min_f0 = 44100/704 ≈ 63Hz
        self.frame_len = self.hop * 8

        # Pre-compute A-weighting curve for perceptual loudness
        self._a_weight = self._build_a_weighting(self.frame_len, sample_rate)

        # Pre-compute mel filterbank matrix (vectorized MFCC)
        self._mel_fb = self._build_mel_filterbank(
            self.frame_len, sample_rate, n_mels=26,
        )

        # Onset detection state
        self._prev_spectrum: np.ndarray | None = None
        self._onset_threshold = 0.3

        # F0 history for vibrato detection
        self._f0_history: list[float] = []
        self._f0_history_max = 50  # ~200ms at 250Hz

    def encode_block(
        self,
        audio: np.ndarray,
        start_time_ms: float = 0.0,
    ) -> list[IntentFrame]:
        """Extract intent frames from an audio block.

        Parameters
        ----------
        audio : np.ndarray, shape (N,), float32, mono, [-1, 1]
        start_time_ms : Timestamp of the first sample in this block.

        Returns
        -------
        List of IntentFrame, one per 4ms frame.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio (1D), got shape {audio.shape}")

        n_frames = max(0, (len(audio) - self.frame_len) // self.hop + 1)
        frames: list[IntentFrame] = []

        for i in range(n_frames):
            start = i * self.hop
            frame_audio = audio[start: start + self.frame_len]
            ts = start_time_ms + (start / self.sample_rate) * 1000.0

            intent = self._extract_frame(frame_audio, ts)
            frames.append(intent)

        return frames

    def _extract_frame(self, frame: np.ndarray, ts: float) -> IntentFrame:
        """Extract all features from a single audio frame."""
        frame64 = frame.astype(np.float64)

        # Sanitize: replace NaN/Inf with zero to prevent corrupt features
        if not np.all(np.isfinite(frame64)):
            frame64 = np.nan_to_num(frame64, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. F0 via YIN
        f0_hz, f0_conf = self._yin_f0(frame64)

        # 2. Loudness — dual measurement:
        #    loudness_db:   time-domain RMS (for decoder gain matching)
        #    loudness_norm: A-weighted (for perceptual display/silence detection)
        rms = float(np.sqrt(np.mean(frame64 ** 2) + 1e-10))
        loudness_db = 20.0 * math.log10(rms + 1e-10)

        # A-weighted perceptual loudness for normalised display
        spectrum_raw = np.abs(np.fft.rfft(frame64))
        a_weighted_rms = float(np.sqrt(np.mean((spectrum_raw * self._a_weight) ** 2) + 1e-10))
        a_weighted_db = 20.0 * math.log10(a_weighted_rms + 1e-10)
        loudness_norm = min(1.0, max(0.0, (a_weighted_db + 40.0) / 60.0))

        # 3. Spectral features
        spectrum = np.abs(np.fft.rfft(frame64 * np.hanning(len(frame64))))
        freqs = np.fft.rfftfreq(len(frame64), 1.0 / self.sample_rate)

        # Spectral centroid
        total_power = np.sum(spectrum) + 1e-10
        centroid = float(np.sum(freqs * spectrum) / total_power)

        # MFCCs (compact timbre representation)
        mfccs = self._compute_mfcc(spectrum, freqs)

        # 4. Onset detection via spectral flux
        is_onset, onset_strength = self._detect_onset(spectrum)

        # 5. Vibrato detection
        vib_rate, vib_depth = self._detect_vibrato(f0_hz)

        return IntentFrame(
            timestamp_ms=ts,
            f0_hz=f0_hz,
            f0_confidence=f0_conf,
            loudness_db=loudness_db,
            loudness_norm=loudness_norm,
            spectral_centroid_hz=centroid,
            mfcc=mfccs,
            is_onset=is_onset,
            onset_strength=onset_strength,
            vibrato_rate_hz=vib_rate,
            vibrato_depth_cents=vib_depth,
            rms_energy=rms,
        )

    def _yin_f0(self, frame: np.ndarray, threshold: float = 0.2) -> tuple[float, float]:
        """YIN pitch estimator with vectorized difference function.

        Uses the full YIN algorithm (de Cheveigné & Kawahara, 2002):
        1. Vectorized difference function (numpy slicing — no Python inner loop)
        2. Cumulative mean normalised difference (CMND)
        3. Absolute threshold search
        4. Parabolic interpolation for sub-sample accuracy

        Returns (f0_hz, confidence).
        """
        n = len(frame)
        w = n // 2

        min_tau = max(2, int(self.sample_rate / F0_MAX_HZ))
        max_tau = min(w - 2, int(self.sample_rate / F0_MIN_HZ) + 1)
        if max_tau <= min_tau + 2:
            return 0.0, 0.0

        x = frame[:w + max_tau].astype(np.float64)

        # Vectorized difference function: d(tau) = sum_j (x[j] - x[j+tau])^2
        # Each iteration is a single vectorized numpy operation
        diff = np.zeros(max_tau + 1)
        for tau in range(1, max_tau + 1):
            diff[tau] = np.sum((x[:w] - x[tau:tau + w]) ** 2)

        # Cumulative mean normalised difference (CMND) — vectorized
        cum_diff = np.cumsum(diff)
        tau_range = np.arange(max_tau + 1, dtype=np.float64)
        tau_range[0] = 1.0  # avoid div by zero
        cmnd = np.ones(max_tau + 1)
        mask = cum_diff > 1e-15
        cmnd[mask] = diff[mask] * tau_range[mask] / cum_diff[mask]

        # Search for first dip below threshold
        for tau in range(min_tau, max_tau):
            if cmnd[tau] < threshold and cmnd[tau] < cmnd[tau + 1]:
                # Parabolic interpolation on the RAW difference function
                if tau > 0 and tau < max_tau:
                    a, b, c = diff[tau - 1], diff[tau], diff[tau + 1]
                    denom = 2.0 * (2 * b - a - c)
                    delta = (a - c) / denom if abs(denom) > 1e-10 else 0.0
                    tau_refined = tau + delta
                else:
                    tau_refined = float(tau)

                if tau_refined < 1.0:
                    continue

                f0 = self.sample_rate / tau_refined
                if F0_MIN_HZ <= f0 <= F0_MAX_HZ:
                    confidence = max(0.0, min(1.0, 1.0 - cmnd[tau]))
                    return f0, confidence

        return 0.0, 0.0

    def _compute_mfcc(self, spectrum: np.ndarray, _freqs: np.ndarray) -> list[float]:
        """Compute 13 MFCCs from magnitude spectrum using pre-built mel filterbank.

        Uses vectorized matrix multiply instead of O(n_mels × n_freq) Python loop.
        """
        # Apply pre-computed filterbank (matrix multiply)
        mel_energies = self._mel_fb @ spectrum

        # Log and vectorized DCT-II
        mel_log = np.log(mel_energies + 1e-10)
        n_mels = len(mel_log)
        n_mfcc = N_MFCC
        indices = np.arange(n_mfcc)[:, None]
        j_range = np.arange(n_mels)[None, :]
        dct_matrix = np.cos(np.pi * indices * (j_range + 0.5) / n_mels)
        mfccs = dct_matrix @ mel_log

        return mfccs.tolist()

    def _detect_onset(self, spectrum: np.ndarray) -> tuple[bool, float]:
        """Detect onsets via spectral flux (half-wave rectified)."""
        if self._prev_spectrum is None:
            self._prev_spectrum = spectrum.copy()
            return False, 0.0

        # Spectral flux: sum of positive differences
        diff = spectrum - self._prev_spectrum
        flux = float(np.sum(np.maximum(0, diff)))

        # Normalise by frame energy
        energy = float(np.sum(spectrum) + 1e-10)
        norm_flux = flux / energy

        self._prev_spectrum = spectrum.copy()

        is_onset = norm_flux > self._onset_threshold
        return is_onset, norm_flux

    def _detect_vibrato(self, f0_hz: float) -> tuple[float, float]:
        """Detect vibrato rate and depth from F0 history."""
        if f0_hz <= 0:
            self._f0_history.clear()
            return 0.0, 0.0

        self._f0_history.append(f0_hz)
        if len(self._f0_history) > self._f0_history_max:
            self._f0_history.pop(0)

        if len(self._f0_history) < 20:  # Need ~80ms minimum
            return 0.0, 0.0

        # Remove DC (mean pitch) and find oscillation
        f0_arr = np.array(self._f0_history)
        mean_f0 = np.mean(f0_arr)
        f0_detrended = f0_arr - mean_f0

        # FFT of pitch contour to find vibrato rate
        fft = np.abs(np.fft.rfft(f0_detrended))
        freqs = np.fft.rfftfreq(len(f0_detrended), 1.0 / FRAME_RATE_HZ)

        # Vibrato is typically 4-8 Hz
        mask = (freqs >= 3.0) & (freqs <= 12.0)
        if not np.any(mask):
            return 0.0, 0.0

        masked_fft = fft.copy()
        masked_fft[~mask] = 0
        peak_idx = int(np.argmax(masked_fft))

        if masked_fft[peak_idx] < 0.5:  # Minimum vibrato energy
            return 0.0, 0.0

        vib_rate = float(freqs[peak_idx])
        # Depth in cents: 1200 × log2(f_max / f_min)
        f0_max, f0_min = np.max(f0_arr), np.min(f0_arr)
        if f0_min > 0:
            vib_depth = 1200.0 * math.log2(f0_max / f0_min)
        else:
            vib_depth = 0.0

        return vib_rate, vib_depth

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    @staticmethod
    def _build_a_weighting(n_fft: int, sr: int) -> np.ndarray:
        """A-weighting curve for perceptual loudness."""
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        freqs = np.maximum(freqs, 1e-6)
        f2 = freqs ** 2
        ra = (
            (12194.0 ** 2 * f2 ** 2)
            / (
                (f2 + 20.6 ** 2)
                * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                * (f2 + 12194.0 ** 2)
            )
        )
        return ra / (np.max(ra) + 1e-10)

    @staticmethod
    def _build_mel_filterbank(n_fft: int, sr: int, n_mels: int = 26) -> np.ndarray:
        """Pre-compute mel filterbank matrix for vectorized MFCC.

        Returns shape (n_mels, n_fft//2 + 1) filterbank matrix.
        """
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        n_bins = len(freqs)

        mel_lo = 2595.0 * np.log10(1.0 + 0 / 700.0)
        mel_hi = 2595.0 * np.log10(1.0 + sr / 2 / 700.0)
        mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
        hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)

        fb = np.zeros((n_mels, n_bins))
        for m in range(n_mels):
            lo, center, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
            # Rising slope
            up = (freqs >= lo) & (freqs <= center)
            fb[m, up] = (freqs[up] - lo) / (center - lo + 1e-10)
            # Falling slope
            down = (freqs > center) & (freqs <= hi)
            fb[m, down] = (hi - freqs[down]) / (hi - center + 1e-10)

        return fb

    def reset(self) -> None:
        """Reset encoder state for a new stream."""
        self._prev_spectrum = None
        self._f0_history.clear()
