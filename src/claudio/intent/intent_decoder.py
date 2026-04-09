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
from pathlib import Path

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
        model_path: str | None = None,
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

        # DDSP neural backend (optional)
        self._ddsp_encoder = None
        self._ddsp_decoder = None
        self._ddsp_device = None
        self.use_ddsp = False

        if model_path is not None:
            self._load_ddsp_model(model_path)

    def decode_frames(self, frames: list[IntentFrame]) -> np.ndarray:
        """Decode a sequence of IntentFrames into audio.

        Uses DDSP neural synthesis if a model is loaded,
        otherwise falls back to additive synthesis.
        """
        if not frames:
            return np.zeros(0, dtype=np.float32)

        # DDSP path: batch-process all frames through neural model
        if self.use_ddsp:
            return self._decode_ddsp(frames)

        # Fallback path: per-frame additive synthesis
        chunks: list[np.ndarray] = []
        for frame in frames:
            chunk = self._decode_single_frame(frame)
            chunks.append(chunk)
            self._prev_frame = frame

        return np.concatenate(chunks).astype(np.float32)

    def _load_ddsp_model(self, model_path: str) -> None:
        """Load trained GRUEncoder + DDSPDecoder from checkpoint."""
        import torch

        from claudio.forge.model.ddsp_decoder import DDSPDecoder
        from claudio.forge.model.gru_encoder import GRUEncoder

        ckpt_file = Path(model_path)
        if not ckpt_file.exists():
            return  # No model file — stay in fallback mode

        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        latent_dim = checkpoint.get("latent_dim", 128)

        self._ddsp_encoder = GRUEncoder(
            input_dim=2, hidden_dim=64, latent_dim=latent_dim, num_layers=2,
        )
        self._ddsp_decoder = DDSPDecoder(
            latent_dim=latent_dim, n_partials=64,
            sample_rate=self.sample_rate,
        )

        self._ddsp_encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self._ddsp_decoder.load_state_dict(checkpoint["decoder_state_dict"])

        self._ddsp_encoder.eval()
        self._ddsp_decoder.eval()

        # Device selection: MPS → CUDA → CPU
        if torch.backends.mps.is_available():
            self._ddsp_device = torch.device("mps")
        elif torch.cuda.is_available():
            self._ddsp_device = torch.device("cuda")
        else:
            self._ddsp_device = torch.device("cpu")

        self._ddsp_encoder.to(self._ddsp_device)
        self._ddsp_decoder.to(self._ddsp_device)
        self.use_ddsp = True

    def _decode_ddsp(self, frames: list[IntentFrame]) -> np.ndarray:
        """Decode frames using trained DDSP neural synthesis."""
        import torch

        n_frames = len(frames)
        f0_min = 32.7
        f0_max = 2093.0
        lo = np.log2(f0_min)
        hi = np.log2(f0_max)

        # Convert frames to normalized F0 + loudness tensors
        f0_arr = np.zeros(n_frames, dtype=np.float32)
        loud_arr = np.zeros(n_frames, dtype=np.float32)

        for i, f in enumerate(frames):
            if f.f0_hz > f0_min:
                f0_arr[i] = (np.log2(f.f0_hz) - lo) / (hi - lo)
            loud_arr[i] = max(0.0, min(1.0, f.loudness_norm))

        # Shape: (1, T) — single batch
        f0_t = torch.from_numpy(f0_arr).unsqueeze(0).to(self._ddsp_device)
        loud_t = torch.from_numpy(loud_arr).unsqueeze(0).to(self._ddsp_device)

        with torch.no_grad():
            z = self._ddsp_encoder(f0_t, loud_t)         # (1, T, 128)
            audio = self._ddsp_decoder(z, f0_t, loud_t)  # (1, T_audio)

        result = audio.squeeze(0).cpu().numpy().astype(np.float32)

        # Soft-clip
        result = np.tanh(result)
        return result

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
