"""
backend_beats.py — AST (Audio Spectrogram Transformer) Backend

MIT AST model fine-tuned on AudioSet (527 classes).
Originally planned as BEATs — replaced with AST because BEATs
is not available as a standard HuggingFace model.

AST achieves SOTA on AudioSet classification (mAP 0.459 on AS-2M).
Expects 16kHz mono audio.

Requirements: pip install transformers torch
"""
from __future__ import annotations

import time

import numpy as np

from .classifier_backend import (
    AudioClassifierBackend,
    ClassificationResult,
    map_label_to_family,
)

_AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


class BEATsBackend(AudioClassifierBackend):
    """AST (Audio Spectrogram Transformer) audio classification backend.

    Named BEATsBackend for interface consistency, but uses the MIT AST model
    which is freely available on HuggingFace and achieves comparable performance.
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._processor = None
        self._model = None
        self._id2label: dict[int, str] = {}
        self._target_sr = 16_000

    @property
    def name(self) -> str:
        return "AST (Audio Spectrogram Transformer)"

    def load_model(self) -> None:
        """Load AST model from HuggingFace."""
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self._processor = ASTFeatureExtractor.from_pretrained(_AST_MODEL_ID)
        self._model = ASTForAudioClassification.from_pretrained(_AST_MODEL_ID)
        self._model.eval()

        if self._device != "cpu" and torch.backends.mps.is_available():
            self._model = self._model.to("mps")
        elif self._device != "cpu":
            self._model = self._model.to(self._device)

        self._id2label = self._model.config.id2label
        print(f"[AST] Model loaded ({len(self._id2label)} AudioSet classes)")

    def classify(
        self, audio: np.ndarray, sample_rate: int, top_k: int = 5,
    ) -> list[ClassificationResult]:
        import torch

        if self._model is None:
            self.load_model()

        # Resample to 16kHz
        resampled = self._resample(audio, sample_rate, self._target_sr)

        t0 = time.perf_counter()

        # Process through feature extractor
        inputs = self._processor(
            resampled, sampling_rate=self._target_sr,
            return_tensors="pt", padding=True,
        )

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000

        # Get top-K
        sorted_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            label = self._id2label.get(int(idx), f"class_{idx}")
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
