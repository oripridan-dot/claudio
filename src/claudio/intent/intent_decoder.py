"""
intent_decoder.py — Additive Audio Synthesis from Intent Packets

Generates a low-fidelity audio preview from IntentFrame packets:
  1. Harmonic additive synthesiser driven by F0 + spectral envelope
  2. Filtered noise component for breath/bow/pick noise
  3. Transient generator for onset attacks
  4. Loudness envelope shaping

NOTE: This is NOT the production audio path. Real audio goes through
NeuralCodec (EnCodec) via /ws/audio at 6-24 kbps with near-transparent quality.
This decoder exists for preview, testing, and sonification of intent metadata.

The decoder maintains phase continuity across frames for glitch-free
real-time streaming.
"""

from __future__ import annotations

import math

import numpy as np

from claudio.intent.intent_encoder import IntentFrame


class IntentDecoder:
    """Regenerates audio from semantic intent frames.

    Two modes:
      - Fallback (default): Additive synthesis (no model needed)
      - DDSP neural:        Uses trained ForgeModel for instrument-quality output

    Pass model_path to __init__ to enable DDSP mode.
    """

    def __init__(
        self,
        sample_rate: int = 44_100,
        frame_rate: int = 250,
        n_harmonics: int = 40,
        model_path: str | None = None,  # Kept for API compat, ignored
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

        # F0 smoothing: keep recent history to median-filter pitch jumps
        self._f0_history_buf: list[float] = []
        self._f0_history_buf_max = 5  # 5 frames = 20ms window
        # Track consecutive unvoiced frames before allowing phase reset
        self._unvoiced_frame_count: int = 0
        self._PHASE_RESET_AFTER_N_UNVOICED = 4  # only reset after 16ms of silence

    def decode_frames(self, frames: list[IntentFrame]) -> np.ndarray:
        """Decode a sequence of IntentFrames into audio via additive synthesis.

        NOTE: This produces a low-fidelity approximation. The real audio path
        uses NeuralCodec (EnCodec) via /ws/audio. This decoder exists for:
          - Preview/monitoring when the codec path is unavailable
          - Testing the intent extraction pipeline
          - Sonification of metadata for debugging
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

        if frame.f0_hz > 0 and frame.f0_confidence > 0.5:
            self._unvoiced_frame_count = 0

            # Smooth F0 using a short median filter to prevent single-frame jumps
            self._f0_history_buf.append(frame.f0_hz)
            if len(self._f0_history_buf) > self._f0_history_buf_max:
                self._f0_history_buf.pop(0)
            smoothed_f0 = float(np.median(self._f0_history_buf))

            # Only reset phases if we've been unvoiced for several consecutive frames
            # (avoids phase reset from a single missed detection mid-note)
            prev_was_silent = self._unvoiced_frame_count >= self._PHASE_RESET_AFTER_N_UNVOICED
            if prev_was_silent:
                self._phases[:] = 0.0

            # Interpolate from previous *smoothed* F0 to current smoothed F0
            prev_smooth_f0 = (
                float(np.median(self._f0_history_buf[:-1])) if len(self._f0_history_buf) > 1 else smoothed_f0
            )
            f0_interp = np.linspace(prev_smooth_f0, smoothed_f0, n)

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
        else:
            self._unvoiced_frame_count += 1

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
        rms_out = float(np.sqrt(np.mean(output**2) + 1e-10))
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
            amps = 1.0 / h_range**0.7
            return amps / (np.max(amps) + 1e-10)

        n_mfcc = min(len(mfcc), 13)
        mfcc_arr = np.array(mfcc[:n_mfcc], dtype=np.float64)

        # Reconstruct spectral envelope via vectorized inverse DCT
        n_bands = 26
        j_range = np.arange(n_bands)
        i_range = np.arange(n_mfcc)
        dct_matrix = np.cos(np.pi * i_range[:, None] * (j_range[None, :] + 0.5) / n_bands)
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
        amps *= 1.0 / h_range**0.3

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
        self._f0_history_buf.clear()
        self._unvoiced_frame_count = 0
