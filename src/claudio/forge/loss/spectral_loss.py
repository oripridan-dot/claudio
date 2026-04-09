"""
src/loss/spectral_loss.py
==========================
Multi-scale STFT reconstruction loss.

Computes L1 distance between log-magnitude spectrograms at 6 different
FFT resolutions to enforce accuracy at both fine-grained and broadband
levels simultaneously.

Reference: Engel et al. (2020) — DDSP: Differentiable Digital Signal Processing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_FFT_SIZES = [64, 128, 256, 512, 1024, 2048]
_HOP_RATIO = 0.25   # hop = fft_size × 0.25


class MultiScaleSpectralLoss(nn.Module):
    """
    Σ_i  L1(log |STFT_i(y_hat)| , log |STFT_i(y)|)
    """

    def __init__(self, fft_sizes: list[int] = _FFT_SIZES, eps: float = 1e-7) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (B, T) predicted audio
        target : (B, T) target audio

        Returns
        -------
        scalar loss tensor
        """
        # Ensure same length
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        loss = torch.zeros(1, device=pred.device)
        for fft_size in self.fft_sizes:
            if min_len < fft_size:
                continue  # Skip FFT sizes larger than the signal
            hop = max(1, int(fft_size * _HOP_RATIO))
            win = torch.hann_window(fft_size, device=pred.device)

            p = pred.reshape(-1, min_len) if pred.dim() > 1 else pred.unsqueeze(0)
            t = target.reshape(-1, min_len) if target.dim() > 1 else target.unsqueeze(0)

            s_pred = torch.stft(
                p, n_fft=fft_size, hop_length=hop, win_length=fft_size,
                window=win, return_complex=True,
            )
            s_tgt = torch.stft(
                t, n_fft=fft_size, hop_length=hop, win_length=fft_size,
                window=win, return_complex=True,
            )

            log_pred = torch.log(s_pred.abs() + self.eps)
            log_tgt  = torch.log(s_tgt.abs()  + self.eps)
            loss = loss + torch.mean(torch.abs(log_pred - log_tgt))

        return loss / len(self.fft_sizes)
