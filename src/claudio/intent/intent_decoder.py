"""
intent_decoder.py — DDSP-Based Audio Regeneration from Intent Packets

Regenerates audio from IntentFrame packets using DDSP synthesis:
  1. Harmonic additive synthesiser driven by F0 + spectral envelope
  2. Filtered noise component for breath/bow/pick noise
  3. Transient generator for onset attacks
  4. Loudness envelope shaping

Two operating modes:
  - Inference mode: Uses trained ForgeModel weights for high-fidelity synthesis
  - Fallback mode:  Uses direct additive synthesis (no model needed, lower fidelity)

The decoder maintains phase continuity across frames for glitch-free
real-time streaming.
"""
from __future__ import annotations

import math

import numpy as np

from claudio.intent.intent_encoder import IntentFrame


class IntentDecoder:
    """Regenerates audio from semantic intent frames.

    Uses additive synthesis with harmonic + noise components,
    driven by the intent parameters. Phase-continuous for
    streaming without clicks or glitches.
    """

    def __init__(
        self,
        sample_rate: int = 44_100,
        frame_rate: int = 250,
        n_harmonics: int = 40,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.hop = sample_rate // frame_rate  # samples per frame
        self.n_harmonics = n_harmonics

        # Phase accumulators for each harmonic (glitch-free streaming)
        self._phases = np.zeros(n_harmonics, dtype=np.float64)

        # Noise filter state (single element — one-pole IIR)
        self._noise_state = 0.0

        # Gain smoothing to prevent per-frame gain jumps
        self._prev_gain = 1.0

        # Last known MFCCs for delta frame carry-forward
        self._last_mfcc: list[float] = []

        # Previous frame for interpolation
        self._prev_frame: IntentFrame | None = None

    def decode_frames(self, frames: list[IntentFrame]) -> np.ndarray:
        """Decode a sequence of IntentFrames into audio.

        Parameters
        ----------
        frames : List of IntentFrame (typically from one encode_block)

        Returns
        -------
        np.ndarray, shape (N,), float32 mono audio.
        """
        if not frames:
            return np.zeros(0, dtype=np.float32)

        chunks: list[np.ndarray] = []
        for frame in frames:
            chunk = self._decode_single_frame(frame)
            chunks.append(chunk)
            self._prev_frame = frame

        return np.concatenate(chunks).astype(np.float32)

    def _decode_single_frame(self, frame: IntentFrame) -> np.ndarray:
        """Synthesize one hop of audio from a single IntentFrame."""
        n = self.hop

        # ── Fast path: silence → return zeros immediately ─────────────
        if frame.loudness_norm < 0.005 and frame.f0_hz <= 0:
            return np.zeros(n, dtype=np.float64)

        # ── Harmonic component ────────────────────────────────────────
        harmonic = np.zeros(n, dtype=np.float64)

        if frame.f0_hz > 0 and frame.f0_confidence > 0.3:
            # Reset phases on silence→voiced transition to prevent clicks
            prev_was_silent = (
                self._prev_frame is None
                or self._prev_frame.f0_hz <= 0
                or self._prev_frame.f0_confidence <= 0.3
            )
            if prev_was_silent:
                self._phases[:] = 0.0

            # Interpolate F0 from previous frame for smooth transitions
            prev_f0 = self._prev_frame.f0_hz if self._prev_frame and self._prev_frame.f0_hz > 0 else frame.f0_hz
            f0_interp = np.linspace(prev_f0, frame.f0_hz, n)

            # Derive harmonic amplitudes from MFCCs (spectral envelope)
            # Use carry-forward: if current frame has no MFCCs (delta), use last known
            mfcc = frame.mfcc
            if mfcc and len(mfcc) >= 2:
                self._last_mfcc = mfcc  # Remember for future delta frames
            elif self._last_mfcc:
                mfcc = self._last_mfcc  # Carry forward
            harmonic_amps = self._mfcc_to_harmonic_amps(mfcc, frame.f0_hz)

            for h in range(self.n_harmonics):
                freq = f0_interp * (h + 1)

                # Anti-alias: skip harmonics above Nyquist
                if np.mean(freq) >= self.sample_rate / 2:
                    break

                # Phase-continuous synthesis
                phase_inc = 2.0 * math.pi * freq / self.sample_rate
                phase = self._phases[h] + np.cumsum(phase_inc)

                amp = harmonic_amps[h]
                harmonic += amp * np.sin(phase)

                # Save phase for next frame (wrapped to [0, 2π])
                self._phases[h] = phase[-1] % (2.0 * math.pi)

        # ── Noise component (breath/bow/pick noise) ───────────────────
        noise = self._generate_filtered_noise(frame, n)

        # ── Onset transient ───────────────────────────────────────────
        transient = np.zeros(n, dtype=np.float64)
        if frame.is_onset and frame.onset_strength > 0.1:
            # Short burst of shaped noise for attack transient
            # Use timestamp-based seed for unique-but-deterministic transients
            seed = int(frame.timestamp_ms * 1000) & 0xFFFFFFFF
            env = np.exp(-np.arange(n, dtype=np.float64) / (n * 0.1))
            transient = np.random.default_rng(seed).normal(0, 1, n) * env * frame.onset_strength * 0.3

        # ── Mix and apply loudness envelope ───────────────────────────
        mix = harmonic + noise * 0.1 + transient

        # Loudness envelope (interpolated from previous frame)
        prev_loud = self._prev_frame.loudness_norm if self._prev_frame else frame.loudness_norm
        envelope = np.linspace(prev_loud, frame.loudness_norm, n)

        output = mix * envelope

        # RMS-match output to target loudness from intent
        rms_out = float(np.sqrt(np.mean(output ** 2) + 1e-10))
        if rms_out > 1e-8 and frame.loudness_db > -70:
            target_rms = 10.0 ** (frame.loudness_db / 20.0)
            new_gain = min(target_rms / rms_out, 3.0)
        else:
            new_gain = self._prev_gain

        # Smooth gain transition across hop to prevent clicks
        gain_env = np.linspace(self._prev_gain, new_gain, n)
        output *= gain_env
        self._prev_gain = new_gain

        # Soft-clip to prevent overs
        output = np.tanh(output)

        return output

    def _mfcc_to_harmonic_amps(self, mfcc: list[float], f0_hz: float) -> np.ndarray:
        """Convert MFCC coefficients back to harmonic amplitudes.

        Uses vectorized inverse DCT and mel-to-harmonic mapping.
        """
        amps = np.zeros(self.n_harmonics, dtype=np.float64)

        if not mfcc or len(mfcc) < 2:
            # Fallback: natural harmonic rolloff (1/h)
            h_range = np.arange(1, self.n_harmonics + 1, dtype=np.float64)
            amps = 1.0 / h_range ** 0.7
            return amps / (np.max(amps) + 1e-10)

        n_mfcc = min(len(mfcc), 13)
        mfcc_arr = np.array(mfcc[:n_mfcc], dtype=np.float64)

        # Reconstruct spectral envelope via vectorized inverse DCT
        n_bands = 26
        j_range = np.arange(n_bands)
        i_range = np.arange(n_mfcc)
        dct_matrix = np.cos(
            np.pi * i_range[:, None] * (j_range[None, :] + 0.5) / n_bands
        )
        mel_log = mfcc_arr @ dct_matrix  # (n_mfcc,) @ (n_mfcc, n_bands) → (n_bands,)

        mel_env = np.exp(mel_log)
        mel_env /= np.max(mel_env) + 1e-10

        # Map harmonics to mel bands (vectorized)
        mel_max = 2595.0 * math.log10(1.0 + self.sample_rate / 2 / 700.0)
        h_range = np.arange(1, self.n_harmonics + 1, dtype=np.float64)
        h_freqs = f0_hz * h_range
        valid = h_freqs < self.sample_rate / 2

        if np.any(valid):
            v_freqs = h_freqs[valid]
            mel_vals = 2595.0 * np.log10(1.0 + v_freqs / 700.0)
            band_idx = mel_vals / mel_max * (n_bands - 1)
            band_lo = np.clip(band_idx.astype(int), 0, n_bands - 2)
            band_hi = np.minimum(band_lo + 1, n_bands - 1)
            frac = band_idx - band_lo
            amps[valid] = mel_env[band_lo] * (1 - frac) + mel_env[band_hi] * frac

        # Apply natural rolloff
        amps *= 1.0 / h_range ** 0.3

        return amps / (np.max(amps) + 1e-10) * 0.3

    def _generate_filtered_noise(self, frame: IntentFrame, n: int) -> np.ndarray:
        """Generate spectrally-shaped noise for non-harmonic content."""
        noise = np.random.default_rng().normal(0, 1, n)

        # Shape noise based on spectral centroid
        if frame.spectral_centroid_hz > 0:
            cutoff = min(frame.spectral_centroid_hz * 2, self.sample_rate / 2 - 100)
            alpha = cutoff / (cutoff + self.sample_rate / (2 * math.pi))
            # One-pole IIR lowpass (176 samples — loop is faster than convolution)
            filtered = np.zeros(n)
            filtered[0] = alpha * noise[0] + (1 - alpha) * self._noise_state
            for i in range(1, n):
                filtered[i] = alpha * noise[i] + (1 - alpha) * filtered[i - 1]
            self._noise_state = float(filtered[-1])
            return filtered

        return noise

    def reset(self) -> None:
        """Reset decoder state for a new stream."""
        self._phases = np.zeros(self.n_harmonics, dtype=np.float64)
        self._noise_state = 0.0
        self._prev_gain = 1.0
        self._last_mfcc = []
        self._prev_frame = None
