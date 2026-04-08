"""
src/model/gru_encoder.py
=========================
GRU-based temporal encoder.

Receives (F0, loudness) conditioning at 250 Hz and produces a latent
vector z ∈ ℝ^{128} per frame, representing the timbre state of the sound.

Two GRU layers + layer-norm provide stable long-horizon temporal context
(up to ~1.5 s at 250 Hz).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """
    Bidirectional GRU → linear projection → layer norm.

    Input:  (B, T, 2)    — [f0, loudness] conditioning per frame
    Output: (B, T, 128)  — latent timbre vector per frame
    """

    def __init__(
        self,
        input_dim:  int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, f0: torch.Tensor, loudness: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        f0        : (B, T) normalised fundamental frequency
        loudness  : (B, T) normalised RMS loudness

        Returns
        -------
        z : (B, T, latent_dim)
        """
        x = torch.stack([f0, loudness], dim=-1)    # (B, T, 2)
        x = torch.tanh(self.input_proj(x))         # (B, T, hidden)
        h, _ = self.gru(x)                         # (B, T, hidden)
        z = self.norm(self.out_proj(h))             # (B, T, latent_dim)
        return z
