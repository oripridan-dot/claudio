"""
classifier_backend.py — Abstract Neural Audio Classification Interface

Defines the common interface for all neural audio classification backends
(PANNs, CLAP, BEATs). Each backend implements `classify()` which maps
raw audio to a ranked list of ClassificationResult.

The top-level InstrumentClassifier uses this interface to delegate
primary classification to a neural model, falling back to heuristics
when neural confidence is below threshold.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .instrument_classifier import InstrumentFamily

# ─── AudioSet → InstrumentFamily Mapping ─────────────────────────────────────
#
# AudioSet has 527 classes. We map the musically-relevant ones to
# Claudio's InstrumentFamily taxonomy. Unmapped labels get UNKNOWN.

AUDIOSET_TO_FAMILY: dict[str, InstrumentFamily] = {
    # Electric guitar
    "Electric guitar": InstrumentFamily.GUITAR_ELECTRIC,
    "Guitar": InstrumentFamily.GUITAR_ELECTRIC,
    "Distortion": InstrumentFamily.GUITAR_ELECTRIC,
    "Power tool": InstrumentFamily.UNKNOWN,  # false positive trap
    # Acoustic guitar
    "Acoustic guitar": InstrumentFamily.GUITAR_ACOUSTIC,
    "Steel guitar, slide guitar": InstrumentFamily.GUITAR_ACOUSTIC,
    "Classical guitar": InstrumentFamily.GUITAR_ACOUSTIC,
    # Bass
    "Bass guitar": InstrumentFamily.BASS_ELECTRIC,
    "Bass drum": InstrumentFamily.DRUMS_KICK,
    # Drums
    "Drum kit": InstrumentFamily.DRUMS_SNARE,
    "Drum": InstrumentFamily.DRUMS_SNARE,
    "Snare drum": InstrumentFamily.DRUMS_SNARE,
    "Rimshot": InstrumentFamily.DRUMS_SNARE,
    "Hi-hat": InstrumentFamily.DRUMS_HIHAT,
    "Cymbal": InstrumentFamily.DRUMS_CYMBAL,
    "Crash cymbal": InstrumentFamily.DRUMS_CYMBAL,
    "Splash, plop": InstrumentFamily.DRUMS_CYMBAL,
    "Tom-tom drum": InstrumentFamily.DRUMS_TOM,
    # Vocals
    "Singing": InstrumentFamily.VOCAL_FEMALE,
    "Male singing": InstrumentFamily.VOCAL_MALE,
    "Female singing": InstrumentFamily.VOCAL_FEMALE,
    "Choir": InstrumentFamily.VOCAL_FEMALE,
    "Voice": InstrumentFamily.VOCAL_MALE,
    "Speech": InstrumentFamily.VOCAL_MALE,
    # Keys
    "Piano": InstrumentFamily.KEYS_PIANO,
    "Electric piano": InstrumentFamily.KEYS_PIANO,
    "Keyboard (musical)": InstrumentFamily.KEYS_SYNTH,
    "Synthesizer": InstrumentFamily.KEYS_SYNTH,
    "Organ": InstrumentFamily.KEYS_ORGAN,
    "Hammond organ": InstrumentFamily.KEYS_ORGAN,
    # Brass
    "Trumpet": InstrumentFamily.BRASS,
    "Trombone": InstrumentFamily.BRASS,
    "French horn": InstrumentFamily.BRASS,
    "Brass instrument": InstrumentFamily.BRASS,
    # Woodwind
    "Saxophone": InstrumentFamily.WOODWIND,
    "Flute": InstrumentFamily.WOODWIND,
    "Clarinet": InstrumentFamily.WOODWIND,
    "Oboe": InstrumentFamily.WOODWIND,
    # Strings
    "Violin, fiddle": InstrumentFamily.STRINGS,
    "Cello": InstrumentFamily.STRINGS,
    "Double bass": InstrumentFamily.BASS_ACOUSTIC,
    "Harp": InstrumentFamily.STRINGS,
    "String section": InstrumentFamily.STRINGS,
}


def map_label_to_family(label: str) -> InstrumentFamily:
    """Map an AudioSet / CLAP label to InstrumentFamily."""
    # Exact match
    if label in AUDIOSET_TO_FAMILY:
        return AUDIOSET_TO_FAMILY[label]
    # Case-insensitive partial match
    label_lower = label.lower()
    for key, family in AUDIOSET_TO_FAMILY.items():
        if key.lower() in label_lower or label_lower in key.lower():
            return family
    return InstrumentFamily.UNKNOWN


# ─── Classification Result ───────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """A single classification from a neural backend."""
    label: str                         # raw model output label
    confidence: float                  # 0.0-1.0
    family: InstrumentFamily           # mapped to Claudio taxonomy
    latency_ms: float = 0.0           # time taken for inference


@dataclass
class BenchmarkResult:
    """Performance benchmark for a single backend."""
    backend_name: str
    results: list[ClassificationResult]
    total_latency_ms: float
    memory_mb: float = 0.0
    top1_label: str = ""
    top1_confidence: float = 0.0
    top1_family: InstrumentFamily = InstrumentFamily.UNKNOWN
    metadata: dict = field(default_factory=dict)


# ─── Abstract Backend ────────────────────────────────────────────────────────

class AudioClassifierBackend(ABC):
    """Abstract interface for neural audio classification backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend."""

    @abstractmethod
    def load_model(self) -> None:
        """Download and load the model. Called once at startup."""

    @abstractmethod
    def classify(
        self, audio: np.ndarray, sample_rate: int, top_k: int = 5,
    ) -> list[ClassificationResult]:
        """
        Classify audio and return top-K results.

        Args:
            audio: 1D float32 array, mono, any sample rate
            sample_rate: sample rate of the audio
            top_k: number of top results to return

        Returns:
            List of ClassificationResult sorted by confidence (descending)
        """

    def benchmark(
        self, audio: np.ndarray, sample_rate: int, n_runs: int = 10,
    ) -> BenchmarkResult:
        """Run inference multiple times and return timing stats."""
        latencies = []
        last_results: list[ClassificationResult] = []

        for _ in range(n_runs):
            t0 = time.perf_counter()
            last_results = self.classify(audio, sample_rate, top_k=5)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        top = last_results[0] if last_results else None

        return BenchmarkResult(
            backend_name=self.name,
            results=last_results,
            total_latency_ms=avg_latency,
            top1_label=top.label if top else "",
            top1_confidence=top.confidence if top else 0.0,
            top1_family=top.family if top else InstrumentFamily.UNKNOWN,
        )
