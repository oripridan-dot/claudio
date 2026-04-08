"""Smoke test for the DDSP forge pipeline."""
import pytest
import torch

from claudio.forge.model.ddsp_decoder import DDSPDecoder
from claudio.forge.model.gru_encoder import GRUEncoder
from claudio.forge.model.forge_model import ForgeModel
from claudio.forge.loss.spectral_loss import MultiScaleSpectralLoss


def test_gru_encoder_shape():
    encoder = GRUEncoder(input_dim=2, hidden_dim=64, latent_dim=128)
    f0 = torch.randn(2, 100)
    loudness = torch.randn(2, 100)
    z = encoder(f0, loudness)
    assert z.shape == (2, 100, 128)


def test_ddsp_decoder_output_shape():
    decoder = DDSPDecoder(latent_dim=128, n_partials=64, sample_rate=44100, frame_rate=250)
    z = torch.randn(1, 50, 128)
    f0 = torch.rand(1, 50)
    loudness = torch.rand(1, 50)
    audio = decoder(z, f0, loudness)
    expected_len = 50 * (44100 // 250)
    assert audio.shape == (1, expected_len)


def test_forge_model_roundtrip():
    model = ForgeModel(sample_rate=44100, n_partials=32, latent_dim=64, gru_hidden=32, gru_layers=1)
    audio_in = torch.randn(1, 44100)
    audio_out = model(audio_in)
    assert audio_out.shape == audio_in.shape


def test_spectral_loss_scalar():
    loss_fn = MultiScaleSpectralLoss()
    pred = torch.randn(1, 8000)
    target = torch.randn(1, 8000)
    loss = loss_fn(pred, target)
    assert loss.dim() == 1
    assert loss.item() > 0
