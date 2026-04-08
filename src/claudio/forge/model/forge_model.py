"""
src/model/forge_model.py
=========================
Full Claudio Analog Forge model assembly.

ForgeModel = FeatureExtractor → GRUEncoder → DDSPDecoder

This is the top-level nn.Module used for training and ONNX export.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from claudio.forge.model.ddsp_decoder import DDSPDecoder
from claudio.forge.model.feature_extractor import FeatureExtractor
from claudio.forge.model.gru_encoder import GRUEncoder


class ForgeModel(nn.Module):
    """
    End-to-end differentiable audio synthesiser.

    Input:  raw audio waveform (B, T), mono, 44 100 Hz
    Output: resynthesised audio (B, T)
    """

    def __init__(
        self,
        sample_rate: int = 44_100,
        n_partials:  int = 64,
        latent_dim:  int = 128,
        gru_hidden:  int = 64,
        gru_layers:  int = 2,
    ) -> None:
        super().__init__()
        self.extractor = FeatureExtractor(sample_rate=sample_rate)
        self.encoder   = GRUEncoder(
            input_dim=2,
            hidden_dim=gru_hidden,
            latent_dim=latent_dim,
            num_layers=gru_layers,
        )
        self.decoder = DDSPDecoder(
            latent_dim=latent_dim,
            n_partials=n_partials,
            sample_rate=sample_rate,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        f0, loudness   = self.extractor(audio)          # (B, T_f)
        z              = self.encoder(f0, loudness)      # (B, T_f, latent)
        audio_out      = self.decoder(z, f0, loudness)   # (B, T_audio)

        # Trim / pad to match input length
        T_in = audio.shape[-1]
        T_out = audio_out.shape[-1]
        if T_out > T_in:
            audio_out = audio_out[..., :T_in]
        elif T_out < T_in:
            audio_out = torch.nn.functional.pad(audio_out, (0, T_in - T_out))

        return audio_out
