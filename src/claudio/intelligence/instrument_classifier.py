"""
instrument_classifier.py — Acoustic Instrument & Model Classifier

Combines spectral fingerprint, transient profile, and harmonic structure
to classify the instrument family and pickup type from audio alone.

For model-level identification (e.g. "Fender Telecaster"), this classifier
returns candidate hints that are cross-validated by MultimodalFusion
with camera data.

Supports optional neural backend (PANNs/CLAP/BEATs) for primary
classification with heuristic fallback when neural confidence is low.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .harmonic_profiler import HarmonicProfile, HarmonicProfiler
from .pickup_detector import PickupDetector, PickupType
from .spectral_extractor import SpectralExtractor, SpectralFingerprint
from .transient_analyzer import TransientAnalyzer, TransientProfile

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


@dataclass
class InstrumentDetection:
    """Result of instrument classification."""
    family: InstrumentFamily
    confidence: float                      # 0.0-1.0
    pickup_type: PickupType = PickupType.UNKNOWN
    model_guess: str = ""                  # e.g. "Fender Stratocaster" (from fusion with vision)
    model_confidence: float = 0.0
    spectral_fingerprint: SpectralFingerprint | None = None
    transient_profile: TransientProfile | None = None
    harmonic_profile: HarmonicProfile | None = None
    coaching_hints: list[str] = field(default_factory=list)
    neural_label: str = ""                 # raw neural model output label
    neural_confidence: float = 0.0         # neural model confidence
    classification_source: str = "heuristic"  # "heuristic", "neural", "fused"


# ─── Main Classifier ─────────────────────────────────────────────────────────


class InstrumentClassifier:
    """
    Full-pipeline instrument classifier.

    Combines spectral fingerprint, transient profile, and harmonic structure
    to classify the instrument family and pickup type from audio alone.

    Supports an optional neural backend for higher-accuracy classification.
    When a neural backend is available:
      1. Neural model runs first and produces a primary classification
      2. Heuristic classification runs in parallel
      3. Results are fused — high-confidence neural results take priority,
         low-confidence neural results are cross-validated with heuristics
    """

    # Import lazily to avoid circular deps
    NEURAL_CONFIDENCE_THRESHOLD = 0.6  # below this, fall back to heuristic

    def __init__(
        self,
        sample_rate: int = 48_000,
        neural_backend: object | None = None,
    ):
        self._sr = sample_rate
        self._spectral = SpectralExtractor(sample_rate)
        self._transient = TransientAnalyzer(sample_rate)
        self._harmonic = HarmonicProfiler(sample_rate)
        self._pickup = PickupDetector()
        self._neural_backend = neural_backend

    def set_neural_backend(self, backend: object) -> None:
        """Hot-swap the neural classification backend."""
        self._neural_backend = backend

    def classify(self, audio: np.ndarray) -> InstrumentDetection:
        # Always run heuristic analysis (provides spectral features regardless)
        fp = self._spectral.extract(audio)
        tp = self._transient.analyze(audio)
        hp = self._harmonic.analyze(audio)

        heuristic_family, heuristic_confidence = self._classify_family(fp, tp, hp)

        # Neural classification (if backend available)
        neural_label = ""
        neural_confidence = 0.0
        neural_family: InstrumentFamily | None = None
        source = "heuristic"

        if self._neural_backend is not None:
            try:
                results = self._neural_backend.classify(audio, self._sr)
                if results:
                    top = results[0]
                    neural_label = top.label
                    neural_confidence = top.confidence
                    neural_family = top.family
            except Exception:
                pass  # neural backend failure → silent fallback

        # Fuse neural + heuristic
        if neural_family is not None and neural_confidence >= self.NEURAL_CONFIDENCE_THRESHOLD:
            family = neural_family
            confidence = neural_confidence
            source = "neural"
            # If heuristic agrees, boost confidence
            if heuristic_family == neural_family:
                confidence = min(1.0, neural_confidence + heuristic_confidence * 0.2)
                source = "fused"
        else:
            family = heuristic_family
            confidence = heuristic_confidence

        # Pickup detection for electric instruments
        pickup = PickupType.UNKNOWN
        if family in (InstrumentFamily.GUITAR_ELECTRIC, InstrumentFamily.BASS_ELECTRIC):
            pickup, _ = self._pickup.classify(fp)

        hints = self._generate_coaching_hints(family, fp, tp, hp)

        return InstrumentDetection(
            family=family,
            confidence=confidence,
            pickup_type=pickup,
            spectral_fingerprint=fp,
            transient_profile=tp,
            harmonic_profile=hp,
            coaching_hints=hints,
            neural_label=neural_label,
            neural_confidence=neural_confidence,
            classification_source=source,
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

        # Guitar electric
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

        # Guitar acoustic
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

        # Bass electric
        be_score = 0.0
        if 30 < hp.fundamental_hz < 250:
            be_score += 0.5
        if fp.harmonic_ratio > 0.4:
            be_score += 0.2
        if fp.spectral_centroid_hz < 1500:
            be_score += 0.3
        scores[InstrumentFamily.BASS_ELECTRIC] = be_score

        # Vocal
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

        # Piano
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
