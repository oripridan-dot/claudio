"""
test_neural_codec.py — Tests for Claudio's NeuralCodec (EnCodec wrapper)

Verifies:
  - Encode/decode round-trip produces valid audio
  - Bandwidth is within expected range
  - Serialization/deserialization works
  - Resampling from 48kHz → 24kHz → 48kHz works
"""

import numpy as np
import pytest

from claudio.codec.neural_codec import CodecFrame, NeuralCodec


@pytest.fixture(scope="module")
def codec():
    """Shared codec instance (model loading is expensive)."""
    return NeuralCodec(bandwidth_kbps=6.0, device="cpu")


@pytest.fixture
def test_signal():
    """2-second sine wave at 440Hz, 48kHz sample rate."""
    sr = 48_000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    return np.sin(2 * np.pi * 440 * t).astype(np.float32), sr


def test_encode_returns_codec_frame(codec, test_signal):
    audio, sr = test_signal
    frame = codec.encode(audio, input_sr=sr)
    assert isinstance(frame, CodecFrame)
    assert frame.codes.ndim == 2
    assert frame.codes.shape[0] == codec.n_codebooks


def test_decode_returns_audio(codec, test_signal):
    audio, sr = test_signal
    frame = codec.encode(audio, input_sr=sr)
    decoded = codec.decode(frame, target_sr=sr)
    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == np.float32
    assert len(decoded) > 0


def test_roundtrip_quality(codec, test_signal):
    """Verify encode→decode doesn't destroy the signal."""
    audio, sr = test_signal
    frame = codec.encode(audio, input_sr=sr)
    decoded = codec.decode(frame, target_sr=sr)

    min_len = min(len(audio), len(decoded))
    error = audio[:min_len] - decoded[:min_len]
    rms_error_db = 20 * np.log10(np.sqrt(np.mean(error ** 2)) + 1e-10)

    # EnCodec should produce reasonable quality (better than -5 dBFS error)
    assert rms_error_db < -5.0, f"RMS error too high: {rms_error_db:.1f} dBFS"


def test_bandwidth_compression(codec, test_signal):
    """Verify codec actually compresses."""
    audio, sr = test_signal
    frame = codec.encode(audio, input_sr=sr)

    raw_bytes = len(audio) * 4  # float32
    codec_bytes = frame.byte_size
    compression_ratio = raw_bytes / codec_bytes

    # Should be at least 10× compression
    assert compression_ratio > 10, f"Compression ratio too low: {compression_ratio:.1f}×"


def test_serialization_roundtrip(codec, test_signal):
    """Verify to_bytes/from_bytes preserves codes."""
    audio, sr = test_signal
    frame = codec.encode(audio, input_sr=sr)

    serialized = frame.to_bytes()
    restored = CodecFrame.from_bytes(serialized, n_codebooks=codec.n_codebooks)

    np.testing.assert_array_equal(frame.codes, restored.codes)


def test_mono_input_required(codec):
    """Verify stereo input is rejected."""
    stereo = np.zeros((2, 1000), dtype=np.float32)
    with pytest.raises(ValueError, match="mono"):
        codec.encode(stereo, input_sr=48_000)


def test_silence_roundtrip(codec):
    """Verify silence survives encode/decode."""
    silence = np.zeros(48_000, dtype=np.float32)
    frame = codec.encode(silence, input_sr=48_000)
    decoded = codec.decode(frame, target_sr=48_000)

    # Output should be near-silence
    rms = np.sqrt(np.mean(decoded ** 2))
    assert rms < 0.01, f"Decoded silence too loud: RMS={rms:.4f}"
