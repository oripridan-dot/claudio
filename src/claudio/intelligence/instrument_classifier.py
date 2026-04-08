"""
instrument_classifier.py — Acoustic Instrument & Model Classifier

Analyses audio streams to identify instruments down to the make/model level
by extracting acoustic signatures: timbre fingerprint, spectral envelope,
transient profile, and harmonic series.

Combined with vision data via MultimodalFusion, this achieves high-confidence
instrument + model detection (e.g. "Fender Telecaster, bridge pickup").

Architecture:
  - SpectralFingerprint: mel-spectrogram + MFCC + spectral centroid/rolloff
  - TransientAnalyzer: attack time, decay shape, pluck/strike classification
  - HarmonicProfiler: partial series analysis, inharmonicity coefficient
  - InstrumentClassifier: fuzzy classification with confidence scoring
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ─── Instrument Taxonomy ──────────────────────────────────────────────────────

class InstrumentFamily(Enum):
    GUITAR_ELECTRIC = "guitar_electric"
    GUITAR_ACOUSTIC = "guitar_acoustic"
    BASS_ELECTRIC = "bass_electric"
    BASS_ACOUSTIC = "bass_acoustic"
    DRUMS_KICK = "drums_kick"
    DRUMS_SNARE = "drums_snare"
    DRUMS_HIHAT = "drums_hihat"
    DRUMS_CYMBAL = "drums_cymbal"
    DRUMS_TOM = "drums_tom"
    VOCAL_MALE = "vocal_male"
    VOCAL_FEMALE = "vocal_female"
    KEYS_PIANO = "keys_piano"
    KEYS_SYNTH = "keys_synth"
    KEYS_ORGAN = "keys_organ"
    BRASS = "brass"
    WOODWIND = "woodwind"
    STRINGS = "strings"
    UNKNOWN = "unknown"


class PickupType(Enum):
    SINGLE_COIL = "single_coil"
    HUMBUCKER = "humbucker"
    P90 = "p90"
    PIEZO = "piezo"
    ACTIVE = "active"
    UNKNOWN = "unknown"


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


@dataclass
class TransientProfile:
    """Attack and decay characteristics of a sound event."""
    attack_time_ms: float                  # time to peak (0→peak)
    decay_time_ms: float                   # time from peak to -20dB
    is_percussive: bool                    # True if attack < 10ms
    is_sustained: bool                     # True if decay > 500ms
    transient_sharpness: float             # 0-1; 1 = extremely sharp like a pick
    attack_frequency_centroid_hz: float    # where the transient energy is concentrated


@dataclass
class HarmonicProfile:
    """Partial series analysis — identifies instrument via overtone structure."""
    fundamental_hz: float
    n_partials: int
    partial_amplitudes: np.ndarray         # relative amplitude of each partial
    inharmonicity_coefficient: float       # 0=perfect harmonic, >0=stiff string (piano/bell)
    odd_even_ratio: float                  # >1 = odd-dominant (clarinet/square), <1 = even-dominant


@dataclass
class InstrumentDetection:
    """Result of instrument classification."""
    family: InstrumentFamily
    confidence: float                      # 0.0-1.0
    pickup_type: PickupType = PickupType.UNKNOWN
    model_guess: str = ""                  # e.g. "Fender Stratocaster" (from fusion with vision)
    model_confidence: float = 0.0
    spectral_fingerprint: Optional[SpectralFingerprint] = None
    transient_profile: Optional[TransientProfile] = None
    harmonic_profile: Optional[HarmonicProfile] = None
    coaching_hints: list[str] = field(default_factory=list)


# ─── Spectral Feature Extraction ─────────────────────────────────────────────

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


# ─── Transient Analyzer ──────────────────────────────────────────────────────

class TransientAnalyzer:
    """Measures attack/decay characteristics to distinguish percussive vs sustained."""

    def __init__(self, sample_rate: int = 48_000):
        self._sr = sample_rate

    def analyze(self, audio: np.ndarray) -> TransientProfile:
        envelope = self._compute_envelope(audio)
        if len(envelope) == 0 or np.max(envelope) < 1e-8:
            return TransientProfile(
                attack_time_ms=0, decay_time_ms=0,
                is_percussive=False, is_sustained=False,
                transient_sharpness=0, attack_frequency_centroid_hz=0,
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
        attack_window = audio[max(0, peak_idx - int(0.005 * self._sr)):
                              peak_idx + int(0.005 * self._sr)]
        if len(attack_window) > 64:
            spectrum = np.abs(np.fft.rfft(attack_window * np.hanning(len(attack_window))))
            freqs = np.fft.rfftfreq(len(attack_window), d=1.0 / self._sr)
            power = spectrum ** 2 + 1e-10
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
        return np.convolve(rectified, kernel, mode='same')


# ─── Harmonic Profiler ───────────────────────────────────────────────────────

class HarmonicProfiler:
    """Analyses the partial series of a pitched instrument to identify timbre."""

    def __init__(self, sample_rate: int = 48_000, n_fft: int = 4096):
        self._sr = sample_rate
        self._n_fft = n_fft

    def analyze(self, audio: np.ndarray, fundamental_hint: float = 0.0) -> HarmonicProfile:
        if len(audio) < self._n_fft:
            audio = np.pad(audio, (0, self._n_fft - len(audio)))

        window = np.hanning(self._n_fft)
        spectrum = np.abs(np.fft.rfft(audio[:self._n_fft] * window))
        freqs = np.fft.rfftfreq(self._n_fft, d=1.0 / self._sr)

        # Find fundamental via autocorrelation (YIN-lite)
        if fundamental_hint <= 0:
            f0 = self._estimate_f0(audio[:self._n_fft])
        else:
            f0 = fundamental_hint

        if f0 < 20:
            return HarmonicProfile(
                fundamental_hz=0, n_partials=0,
                partial_amplitudes=np.array([]),
                inharmonicity_coefficient=0, odd_even_ratio=1.0,
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
        max_lag = int(self._sr / 60)   # 60 Hz minimum
        min_lag = int(self._sr / 4000)  # 4kHz maximum
        if max_lag >= len(audio) // 2:
            max_lag = len(audio) // 2 - 1

        # Difference function
        d = np.zeros(max_lag)
        for tau in range(min_lag, max_lag):
            diff = audio[:len(audio) - tau] - audio[tau:2 * (len(audio) - tau) // 2 + tau]
            if len(diff) > 0:
                d[tau] = np.sum(diff[:len(audio) - max_lag] ** 2)

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


# ─── Pickup/Source Detector ──────────────────────────────────────────────────

class PickupDetector:
    """Classifies electric guitar pickup type from spectral characteristics."""

    # Single-coil: bright (high centroid), narrow bandwidth, prominent 3-6kHz
    # Humbucker: darker, wider, stronger mids, weaker highs
    # P90: midrange bark, moderate brightness
    # Piezo: ultra-bright, thin, high flatness

    def classify(self, fingerprint: SpectralFingerprint) -> tuple[PickupType, float]:
        centroid = fingerprint.spectral_centroid_hz
        flatness = fingerprint.spectral_flatness
        rolloff = fingerprint.spectral_rolloff_hz

        scores: dict[PickupType, float] = {}

        # Single coil: bright, clear
        sc_score = 0.0
        if centroid > 2500:
            sc_score += 0.4
        if rolloff > 8000:
            sc_score += 0.3
        if flatness < 0.3:
            sc_score += 0.3
        scores[PickupType.SINGLE_COIL] = sc_score

        # Humbucker: darker, thicker
        hb_score = 0.0
        if centroid < 2200:
            hb_score += 0.4
        if rolloff < 7000:
            hb_score += 0.3
        if flatness < 0.25:
            hb_score += 0.3
        scores[PickupType.HUMBUCKER] = hb_score

        # P90: midrange presence
        p90_score = 0.0
        if 1800 < centroid < 3000:
            p90_score += 0.5
        if 5000 < rolloff < 9000:
            p90_score += 0.3
        if flatness < 0.3:
            p90_score += 0.2
        scores[PickupType.P90] = p90_score

        # Piezo: very bright, thin
        pz_score = 0.0
        if centroid > 3500:
            pz_score += 0.4
        if flatness > 0.35:
            pz_score += 0.4
        if rolloff > 10000:
            pz_score += 0.2
        scores[PickupType.PIEZO] = pz_score

        best = max(scores, key=scores.__getitem__)
        return best, scores[best]


# ─── Main Classifier ─────────────────────────────────────────────────────────

class InstrumentClassifier:
    """
    Full-pipeline instrument classifier.

    Combines spectral fingerprint, transient profile, and harmonic structure
    to classify the instrument family and pickup type from audio alone.

    For model-level identification (e.g. "Fender Telecaster"), this classifier
    returns candidate hints that are cross-validated by MultimodalFusion
    with camera data.
    """

    def __init__(self, sample_rate: int = 48_000):
        self._sr = sample_rate
        self._spectral = SpectralExtractor(sample_rate)
        self._transient = TransientAnalyzer(sample_rate)
        self._harmonic = HarmonicProfiler(sample_rate)
        self._pickup = PickupDetector()

    def classify(self, audio: np.ndarray) -> InstrumentDetection:
        fp = self._spectral.extract(audio)
        tp = self._transient.analyze(audio)
        hp = self._harmonic.analyze(audio)

        family, confidence = self._classify_family(fp, tp, hp)
        pickup = PickupType.UNKNOWN
        pickup_conf = 0.0

        if family in (InstrumentFamily.GUITAR_ELECTRIC, InstrumentFamily.BASS_ELECTRIC):
            pickup, pickup_conf = self._pickup.classify(fp)

        hints = self._generate_coaching_hints(family, fp, tp, hp)

        return InstrumentDetection(
            family=family,
            confidence=confidence,
            pickup_type=pickup,
            spectral_fingerprint=fp,
            transient_profile=tp,
            harmonic_profile=hp,
            coaching_hints=hints,
        )

    def _classify_family(
        self,
        fp: SpectralFingerprint,
        tp: TransientProfile,
        hp: HarmonicProfile,
    ) -> tuple[InstrumentFamily, float]:
        """Rule-based classification with fuzzy scoring."""
        scores: dict[InstrumentFamily, float] = {}

        # Drums: percussive, low harmonic ratio, noisy
        drums_base = 0.0
        if tp.is_percussive:
            drums_base += 0.5
        if fp.harmonic_ratio < 0.3:
            drums_base += 0.3
        if fp.spectral_flatness > 0.4:
            drums_base += 0.2

        # Drum sub-types by frequency
        if drums_base > 0.5:
            if fp.spectral_centroid_hz < 300:
                scores[InstrumentFamily.DRUMS_KICK] = drums_base * 1.1
            elif fp.spectral_centroid_hz < 2000:
                scores[InstrumentFamily.DRUMS_SNARE] = drums_base * 1.0
            elif fp.spectral_centroid_hz < 5000:
                scores[InstrumentFamily.DRUMS_TOM] = drums_base * 0.8
            else:
                if fp.spectral_flatness > 0.5:
                    scores[InstrumentFamily.DRUMS_CYMBAL] = drums_base * 0.9
                else:
                    scores[InstrumentFamily.DRUMS_HIHAT] = drums_base * 0.9

        # Guitar electric: harmonic, medium attack, bright centroid
        eg_score = 0.0
        if fp.harmonic_ratio > 0.4:
            eg_score += 0.3
        if 80 < hp.fundamental_hz < 1200:
            eg_score += 0.3
        if tp.attack_time_ms < 30:
            eg_score += 0.2
        if fp.spectral_centroid_hz > 1500:
            eg_score += 0.2
        scores[InstrumentFamily.GUITAR_ELECTRIC] = eg_score

        # Guitar acoustic: similar but higher flatness, stronger decay
        ag_score = 0.0
        if fp.harmonic_ratio > 0.35:
            ag_score += 0.25
        if 80 < hp.fundamental_hz < 1200:
            ag_score += 0.25
        if tp.attack_time_ms < 20:
            ag_score += 0.2
        if fp.spectral_flatness > 0.15:
            ag_score += 0.15
        if tp.decay_time_ms > 200:
            ag_score += 0.15
        scores[InstrumentFamily.GUITAR_ACOUSTIC] = ag_score

        # Bass electric: low fundamental, harmonic
        be_score = 0.0
        if 30 < hp.fundamental_hz < 250:
            be_score += 0.5
        if fp.harmonic_ratio > 0.4:
            be_score += 0.2
        if fp.spectral_centroid_hz < 1500:
            be_score += 0.3
        scores[InstrumentFamily.BASS_ELECTRIC] = be_score

        # Vocal: sustained, highly harmonic, wide bandwidth
        vocal_base = 0.0
        if tp.is_sustained:
            vocal_base += 0.3
        if fp.harmonic_ratio > 0.6:
            vocal_base += 0.3
        if fp.spectral_bandwidth_hz > 2000:
            vocal_base += 0.2
        if hp.fundamental_hz > 0 and hp.n_partials > 5:
            vocal_base += 0.2

        if vocal_base > 0.5:
            if hp.fundamental_hz < 200:
                scores[InstrumentFamily.VOCAL_MALE] = vocal_base
            else:
                scores[InstrumentFamily.VOCAL_FEMALE] = vocal_base

        # Piano: percussive but harmonic, high inharmonicity
        piano_score = 0.0
        if tp.is_percussive and fp.harmonic_ratio > 0.5:
            piano_score += 0.4
        if hp.inharmonicity_coefficient > 0.01:
            piano_score += 0.3
        if tp.decay_time_ms > 300:
            piano_score += 0.3
        scores[InstrumentFamily.KEYS_PIANO] = piano_score

        if not scores:
            return InstrumentFamily.UNKNOWN, 0.0

        best = max(scores, key=scores.__getitem__)
        return best, min(1.0, scores[best])

    def _generate_coaching_hints(
        self,
        family: InstrumentFamily,
        fp: SpectralFingerprint,
        tp: TransientProfile,
        hp: HarmonicProfile,
    ) -> list[str]:
        hints: list[str] = []

        if family in (InstrumentFamily.GUITAR_ELECTRIC, InstrumentFamily.GUITAR_ACOUSTIC):
            if tp.transient_sharpness > 0.8:
                hints.append(
                    "High transient sharpness detected — your pick is hitting "
                    "the string at a steep angle, creating a harsh 3-6kHz spike. "
                    "Try angling the pick 10-15° to soften the attack."
                )
            if fp.spectral_centroid_hz > 4000 and family == InstrumentFamily.GUITAR_ELECTRIC:
                hints.append(
                    "Very bright tone detected — if you're on the bridge pickup, "
                    "try rolling the tone knob back to 7 to tame the ice-pick frequencies."
                )

        if family == InstrumentFamily.VOCAL_MALE or family == InstrumentFamily.VOCAL_FEMALE:
            if fp.spectral_centroid_hz < 800:
                hints.append(
                    "Your vocal sounds muffled — check mic proximity. "
                    "Step back 6 inches to reduce proximity effect."
                )

        if family in (InstrumentFamily.DRUMS_SNARE,):
            if fp.spectral_centroid_hz < 1000:
                hints.append(
                    "The snare sounds boxy. Try tightening the bottom head "
                    "or moving the mic closer to the rim for more crack."
                )

        return hints
