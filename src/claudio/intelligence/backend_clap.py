"""
backend_clap.py — CLAP (Contrastive Language-Audio Pretraining) Backend

Zero-shot audio classification via text prompts.
Uses the HuggingFace transformers pipeline for simplicity,
or falls back to the laion-clap package.

CLAP's key advantage: you define classification categories via natural
language text prompts — no fixed label set.

Requirements: pip install transformers laion-clap
"""

from __future__ import annotations

import time

import numpy as np

from .classifier_backend import (
    AudioClassifierBackend,
    ClassificationResult,
    map_label_to_family,
)

# Instrument-specific prompts — much more detailed than generic AudioSet labels
# These prompts leverage CLAP's zero-shot capability for fine-grained detection.
INSTRUMENT_PROMPTS: list[tuple[str, str]] = [
    ("electric guitar with distortion", "Electric guitar"),
    ("clean electric guitar", "Electric guitar"),
    ("electric guitar with single coil pickup", "Electric guitar"),
    ("electric guitar with humbucker", "Electric guitar"),
    ("acoustic guitar strumming", "Acoustic guitar"),
    ("acoustic guitar fingerpicking", "Acoustic guitar"),
    ("classical guitar", "Classical guitar"),
    ("electric bass guitar", "Bass guitar"),
    ("slap bass", "Bass guitar"),
    ("acoustic upright bass", "Double bass"),
    ("kick drum hit", "Bass drum"),
    ("snare drum hit", "Snare drum"),
    ("hi-hat cymbal", "Hi-hat"),
    ("crash cymbal", "Crash cymbal"),
    ("ride cymbal", "Cymbal"),
    ("tom drum", "Tom-tom drum"),
    ("drum kit playing", "Drum kit"),
    ("male voice singing", "Male singing"),
    ("female voice singing", "Female singing"),
    ("piano playing", "Piano"),
    ("electric piano or Rhodes", "Electric piano"),
    ("synthesizer pad", "Synthesizer"),
    ("synthesizer lead", "Synthesizer"),
    ("organ playing", "Organ"),
    ("trumpet playing", "Trumpet"),
    ("saxophone playing", "Saxophone"),
    ("violin playing", "Violin, fiddle"),
    ("cello playing", "Cello"),
    ("flute playing", "Flute"),
    ("clarinet playing", "Clarinet"),
    ("harmonica playing", "Harmonica"),
    ("silence or background noise", "Silence"),
]


class CLAPBackend(AudioClassifierBackend):
    """CLAP zero-shot audio classification backend."""

    def __init__(self, device: str = "cpu", use_transformers: bool = True):
        self._device = device
        self._use_transformers = use_transformers
        self._pipeline = None
        self._clap_model = None
        self._target_sr = 48_000  # CLAP accepts 48kHz

    @property
    def name(self) -> str:
        return "CLAP (Zero-Shot)"

    def load_model(self) -> None:
        """Load CLAP model via transformers pipeline or laion-clap."""
        if self._use_transformers:
            self._load_transformers()
        else:
            self._load_laion_clap()

    def _load_transformers(self) -> None:
        """Load via HuggingFace transformers pipeline."""
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="zero-shot-audio-classification",
                model="laion/clap-htsat-unfused",
                device=self._device if self._device != "cpu" else -1,
            )
            print("[CLAP] Loaded via transformers pipeline (laion/clap-htsat-unfused)")
        except Exception as e:
            print(f"[CLAP] transformers pipeline failed ({e}), trying laion-clap...")
            self._use_transformers = False
            self._load_laion_clap()

    def _load_laion_clap(self) -> None:
        """Load via laion-clap package directly."""
        import laion_clap

        self._clap_model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel="HTSAT-base",
        )
        self._clap_model.load_ckpt()
        print("[CLAP] Loaded via laion-clap (HTSAT-base)")

    def classify(
        self,
        audio: np.ndarray,
        sample_rate: int,
        top_k: int = 5,
    ) -> list[ClassificationResult]:
        if self._pipeline is None and self._clap_model is None:
            self.load_model()

        # Resample to 48kHz
        resampled = self._resample(audio, sample_rate, self._target_sr)

        prompts = [p[0] for p in INSTRUMENT_PROMPTS]
        audioset_labels = [p[1] for p in INSTRUMENT_PROMPTS]

        t0 = time.perf_counter()

        if self._pipeline is not None:
            # transformers pipeline — expects a dict or array
            result = self._pipeline(
                resampled,
                candidate_labels=prompts,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            results = []
            for item in result[:top_k]:
                prompt_idx = prompts.index(item["label"])
                audioset_label = audioset_labels[prompt_idx]
                family = map_label_to_family(audioset_label)
                results.append(
                    ClassificationResult(
                        label=item["label"],
                        confidence=float(item["score"]),
                        family=family,
                        latency_ms=latency_ms,
                    )
                )
            return results
        else:
            # laion-clap direct usage
            import tempfile

            import soundfile as sf
            import torch

            # Write to temp file (laion-clap needs file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, resampled, self._target_sr)
                audio_embed = self._clap_model.get_audio_embedding_from_filelist(
                    x=[f.name],
                )

            text_prompts = [f"This is a sound of {p}" for p in prompts]
            text_embed = self._clap_model.get_text_embedding(text_prompts)

            similarity = torch.tensor(audio_embed) @ torch.tensor(text_embed).T
            probs = torch.softmax(similarity[0], dim=0).numpy()
            latency_ms = (time.perf_counter() - t0) * 1000

            sorted_indices = np.argsort(probs)[::-1][:top_k]
            results = []
            for idx in sorted_indices:
                family = map_label_to_family(audioset_labels[idx])
                results.append(
                    ClassificationResult(
                        label=prompts[idx],
                        confidence=float(probs[idx]),
                        family=family,
                        latency_ms=latency_ms,
                    )
                )
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
