"""Smoke test for the DDSP forge pipeline."""
import tempfile
from pathlib import Path

import numpy as np
import torch

from claudio.forge.loss.spectral_loss import MultiScaleSpectralLoss
from claudio.forge.model.ddsp_decoder import DDSPDecoder
from claudio.forge.model.forge_model import ForgeModel
from claudio.forge.model.gru_encoder import GRUEncoder
from claudio.intent.intent_decoder import IntentDecoder
from claudio.intent.intent_encoder import IntentEncoder


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


def test_training_loop_convergence():
    """Mini training loop should show loss decrease over 3 epochs."""
    encoder = GRUEncoder(input_dim=2, hidden_dim=32, latent_dim=64, num_layers=1)
    decoder = DDSPDecoder(latent_dim=64, n_partials=16, sample_rate=16000, frame_rate=250)
    loss_fn = MultiScaleSpectralLoss(fft_sizes=[64, 128, 256])

    # Synthetic data: 0.5s clip at 16kHz
    sr = 16000
    clip_len = sr // 2
    hop = sr // 250
    n_frames = clip_len // hop

    audio = torch.sin(2 * torch.pi * 440 * torch.arange(clip_len).float() / sr).unsqueeze(0) * 0.3
    f0_norm = torch.full((1, n_frames), 0.6)  # Normalized A4
    loudness = torch.full((1, n_frames), 0.3)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    losses = []
    for _epoch in range(3):
        optimizer.zero_grad()
        z = encoder(f0_norm, loudness)
        pred = decoder(z, f0_norm, loudness)
        # Trim to match
        min_len = min(pred.shape[-1], audio.shape[-1])
        loss = loss_fn(pred[..., :min_len], audio[..., :min_len])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease over 3 epochs
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_checkpoint_save_load():
    """Save and load checkpoint preserves model state."""
    encoder = GRUEncoder(input_dim=2, hidden_dim=32, latent_dim=64, num_layers=1)
    decoder = DDSPDecoder(latent_dim=64, n_partials=16, sample_rate=16000)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "latent_dim": 64,
        "epoch": 1,
        "loss": 3.5,
    }, ckpt_path)

    # Load into fresh models
    enc2 = GRUEncoder(input_dim=2, hidden_dim=32, latent_dim=64, num_layers=1)
    dec2 = DDSPDecoder(latent_dim=64, n_partials=16, sample_rate=16000)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    enc2.load_state_dict(ckpt["encoder_state_dict"])
    dec2.load_state_dict(ckpt["decoder_state_dict"])

    # Outputs should match
    f0 = torch.rand(1, 20)
    loud = torch.rand(1, 20)
    with torch.no_grad():
        z1 = encoder(f0, loud)
        z2 = enc2(f0, loud)
    assert torch.allclose(z1, z2, atol=1e-6)

    Path(ckpt_path).unlink()


def test_ddsp_decoder_integration():
    """IntentDecoder with model_path loads DDSP and produces audio."""
    # Create a mini model and save it
    encoder = GRUEncoder(input_dim=2, hidden_dim=64, latent_dim=128, num_layers=2)
    decoder = DDSPDecoder(latent_dim=128, n_partials=64, sample_rate=44100)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "latent_dim": 128,
    }, ckpt_path)

    # Create IntentDecoder with DDSP model
    dec = IntentDecoder(sample_rate=44100, model_path=ckpt_path)
    assert dec.use_ddsp is True

    # Encode some audio and decode with DDSP
    enc = IntentEncoder(sample_rate=44100)
    tone = (np.sin(2 * np.pi * 440 * np.arange(22050) / 44100) * 0.3).astype(np.float32)
    frames = enc.encode_block(tone)

    audio_out = dec.decode_frames(frames[:50])
    assert len(audio_out) > 0
    assert audio_out.dtype == np.float32

    Path(ckpt_path).unlink()


def test_decoder_fallback_without_model():
    """IntentDecoder without model_path uses additive synthesis."""
    dec = IntentDecoder(sample_rate=44100)
    assert dec.use_ddsp is False

    # Should still work with additive synthesis
    enc = IntentEncoder(sample_rate=44100)
    tone = (np.sin(2 * np.pi * 440 * np.arange(22050) / 44100) * 0.3).astype(np.float32)
    frames = enc.encode_block(tone)

    audio_out = dec.decode_frames(frames[:50])
    rms = float(np.sqrt(np.mean(audio_out ** 2)))
    assert rms > 0.01, f"Fallback decoder produced near-silence: rms={rms}"


def test_decoder_nonexistent_model_path():
    """IntentDecoder with bad model_path falls back gracefully."""
    dec = IntentDecoder(sample_rate=44100, model_path="/nonexistent/model.pt")
    assert dec.use_ddsp is False  # Should NOT enable DDSP

