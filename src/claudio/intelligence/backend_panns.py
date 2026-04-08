"""
backend_panns.py — PANNs (Pre-trained Audio Neural Networks) Backend

CNN14 model trained on AudioSet (527 classes).
Auto-downloads pretrained checkpoint on first use.

Requirements: pip install panns-inference
PANNs expects 32kHz mono audio.
"""
from __future__ import annotations

import time

import numpy as np

from .classifier_backend import (
    AudioClassifierBackend,
    ClassificationResult,
    map_label_to_family,
)


class PANNsBackend(AudioClassifierBackend):
    """PANNs CNN14 audio classification backend."""

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model = None
        self._labels: list[str] = []
        self._target_sr = 32_000

    @property
    def name(self) -> str:
        return "PANNs (CNN14)"

    def load_model(self) -> None:
        """Load PANNs model — auto-downloads checkpoint on first use."""
        from panns_inference import AudioTagging, labels

        self._model = AudioTagging(checkpoint_path=None, device=self._device)
        self._labels = labels
        print(f"[PANNs] Model loaded ({len(self._labels)} AudioSet classes)")

    def classify(
        self, audio: np.ndarray, sample_rate: int, top_k: int = 5,
    ) -> list[ClassificationResult]:
        if self._model is None:
            self.load_model()

        # Resample to 32kHz if needed
        resampled = self._resample(audio, sample_rate, self._target_sr)

        # PANNs expects shape (batch_size, samples)
        if resampled.ndim == 1:
            resampled = resampled[np.newaxis, :]

        t0 = time.perf_counter()
        clipwise_output, _ = self._model.inference(resampled)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Get top-K indices
        probs = clipwise_output[0]
        sorted_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            label = self._labels[idx]
            confidence = float(probs[idx])
            family = map_label_to_family(label)
            results.append(ClassificationResult(
                label=label,
                confidence=confidence,
                family=family,
                latency_ms=latency_ms,
            ))

        return results

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        if src_sr == target_sr:
            return audio.astype(np.float32)
        ratio = target_sr / src_sr
        n_samples = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        return resampled.astype(np.float32)
