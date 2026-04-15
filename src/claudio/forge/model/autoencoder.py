"""
autoencoder.py — DEPRECATED: Replaced by codec.neural_codec.NeuralCodec

The SemanticVocoder was an STFT→ISTFT identity function pretending to be AI.
The AudioAutoEncoder was a wrapper around it with dummy weights.

Both have been replaced by NeuralCodec (EnCodec-based) which provides
actual neural audio compression at 1.5-24 kbps.

This file is kept only for checkpoint loading compatibility during transition.
Use `from claudio.codec.neural_codec import NeuralCodec` instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioAutoEncoder(nn.Module):
    """DEPRECATED: Use claudio.codec.neural_codec.NeuralCodec instead.

    Kept for backward-compatible checkpoint loading only.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_fft = 1024
        self.hop_length = 256
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 3:
            audio = audio.squeeze(1)
        window = torch.hann_window(self.n_fft, device=audio.device)
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=window, return_complex=True, pad_mode='constant')
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        return torch.cat([mag, phase], dim=1)

    def decode(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        F_bins = z.shape[1] // 2
        mag, phase = z[:, :F_bins, :], z[:, F_bins:, :]
        stft_complex = torch.polar(mag, phase)
        window = torch.hann_window(self.n_fft, device=z.device)
        return torch.istft(stft_complex, n_fft=self.n_fft, hop_length=self.hop_length,
                           window=window, length=target_len)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 3:
            audio = audio.squeeze(1)
        target_len = audio.shape[-1]
        z = self.encode(audio)
        return self.decode(z, target_len)
