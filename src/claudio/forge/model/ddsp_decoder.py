"""
src/model/ddsp_decoder.py
==========================
DDSP Decoder: harmonic additive synthesiser + filtered noise component.

Given:
  - z          : (B, T, 128) latent from GRU encoder
  - f0_hz      : (B, T) fundamental frequency in Hz
  - loudness   : (B, T) amplitude envelope [0, 1]

Produces:
  - audio_out  : (B, T_audio) float32 synthesised audio

Signal path:
  z → [Linear → sigmoid] → partial_amplitudes  (B, T, N_partials)
  z → [Linear → sigmoid] → noise_filter_mags   (B, T, N_filter_bins)

  Harmonic component:
    partials = f0_hz × [1, 2, ..., N]
    audio_harm = sum_i(amp_i × sin(2π ∫ partial_i dt))

  Noise component:
    noise = randn(T_audio)
    noise_filtered = FIR(noise, noise_filter_mags)

  Output = (harmonic + noise) × loudness_envelope
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_PARTIALS    = 64
N_FILTER_BINS = 256
SAMPLE_RATE   = 44_100
FRAME_RATE    = 250    # Hz
HOP_SIZE      = SAMPLE_RATE // FRAME_RATE


class DDSPDecoder(nn.Module):

    def __init__(
        self,
        latent_dim:    int = 128,
        n_partials:    int = N_PARTIALS,
        n_filter_bins: int = N_FILTER_BINS,
        sample_rate:   int = SAMPLE_RATE,
        frame_rate:    int = FRAME_RATE,
    ) -> None:
        super().__init__()
        self.n_partials    = n_partials
        self.n_filter_bins = n_filter_bins
        self.sample_rate   = sample_rate
        self.hop           = sample_rate // frame_rate

        # MLP heads
        self.partial_head = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, n_partials), nn.Sigmoid(),
        )
        self.noise_head = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, n_filter_bins), nn.Sigmoid(),
        )

    def forward(
        self,
        z:            torch.Tensor,   # (B, T, latent_dim)
        f0_norm:      torch.Tensor,   # (B, T) in [0, 1]
        loudness:     torch.Tensor,   # (B, T) in [0, 1]
    ) -> torch.Tensor:
        B, T, _ = z.shape

        # Decode amplitudes + filter
        amp    = self.partial_head(z)          # (B, T, N_partials)
        n_mags = self.noise_head(z)            # (B, T, N_filter_bins)

        # Recover F0 in Hz from log-normalised encoding
        lo = math.log2(32.7)
        hi = math.log2(2093.0)
        f0_hz = 2.0 ** (f0_norm * (hi - lo) + lo)   # (B, T)

        # Build harmonic audio
        audio_harm = self._harmonic_synth(amp, f0_hz, B, T)

        # Build filtered noise
        audio_noise = self._filtered_noise(n_mags, B, T)

        # Upsample loudness envelope (B, T) → (B, T_audio)
        T_audio = T * self.hop
        env = F.interpolate(
            loudness.unsqueeze(1),         # (B, 1, T)
            size=T_audio, mode="linear", align_corners=False,
        ).squeeze(1)                        # (B, T_audio)

        return (audio_harm + audio_noise) * env

    # ── Private helpers ─────────────────────────────────────────────────────

    def _harmonic_synth(
        self,
        amp:   torch.Tensor,   # (B, T, N_partials)
        f0_hz: torch.Tensor,   # (B, T)
        B: int,
        T: int,
    ) -> torch.Tensor:
        T_audio = T * self.hop
        device  = amp.device

        # Partial frequencies: each row multiplied by [1, 2, ..., N]
        partials_idx = torch.arange(1, self.n_partials + 1, device=device).float()
        freq_hz = f0_hz.unsqueeze(-1) * partials_idx.unsqueeze(0).unsqueeze(0)
        # Band-limit: zero partials above Nyquist
        mask = (freq_hz < self.sample_rate / 2).float()

        # Upsample amplitude envelopes: (B, T, N) → (B, N, T_audio)
        amp_up = F.interpolate(
            amp.permute(0, 2, 1),       # (B, N, T)
            size=T_audio, mode="linear", align_corners=False,
        )  # (B, N, T_audio)

        # Upsample & integrate instantaneous frequency → phase
        freq_up = F.interpolate(
            freq_hz.permute(0, 2, 1),   # (B, N, T)
            size=T_audio, mode="linear", align_corners=False,
        )  # (B, N, T_audio)
        phase = torch.cumsum(freq_up / self.sample_rate, dim=-1) * 2 * math.pi

        mask_up = F.interpolate(
            mask.permute(0, 2, 1).float(),
            size=T_audio, mode="nearest",
        )
        sinusoids = torch.sin(phase) * amp_up * mask_up   # (B, N, T_audio)
        return sinusoids.sum(dim=1)   # (B, T_audio)

    def _filtered_noise(
        self,
        n_mags: torch.Tensor,   # (B, T, N_filter_bins)
        B: int,
        T: int,
    ) -> torch.Tensor:
        T_audio = T * self.hop
        device  = n_mags.device

        # White noise
        noise = torch.randn(B, T_audio, device=device)

        # Convert magnitude spectrum → minimum-phase FIR via windowed sinc
        # Upsample per-frame filter to per-sample resolution
        mags_up = F.interpolate(
            n_mags.permute(0, 2, 1),    # (B, N_bins, T)
            size=T_audio, mode="linear", align_corners=False,
        )  # (B, N_bins, T_audio)

        # For efficiency: apply a single mean filter (frame-averaged) per batch
        mean_mags = mags_up.mean(-1)   # (B, N_bins)

        # Build symmetric FIR kernel from magnitude spectrum (IFFT)
        # Pad to next power of two for FFT
        fft_size = self.n_filter_bins * 2
        half = torch.zeros(B, fft_size // 2 + 1, device=device, dtype=torch.complex64)
        half.real[:, : self.n_filter_bins] = mean_mags

        kernel = torch.fft.irfft(half, n=fft_size)   # (B, fft_size)
        # Apply Hann window
        win = torch.hann_window(fft_size, device=device)
        kernel = kernel * win.unsqueeze(0)

        # Convolve noise with per-sample kernel (grouped conv trick: B separate filters)
        noise_filtered = F.conv1d(
            noise.unsqueeze(1),            # (B, 1, T_audio)
            kernel.unsqueeze(1),           # (B, 1, fft_size)
            padding=fft_size // 2,
            groups=1,                      # simplified: use mean kernel for all batch
        ).squeeze(1)[:, :T_audio]          # (B, T_audio)

        return noise_filtered * 0.1   # scale noise contribution
