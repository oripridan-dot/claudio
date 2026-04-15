"""
neural_codec.py — Claudio Audio Codec Layer (EnCodec Wrapper)

Provides a clean interface for encoding and decoding audio via Meta's EnCodec
neural codec. This replaces the fake SemanticVocoder and the broken DDSP path.

Two modes:
  - Streaming: encode/decode individual chunks for real-time WebSocket/WebRTC
  - Batch: encode/decode full audio files for testing/offline processing

Supported bandwidths: 1.5, 3.0, 6.0, 12.0, 24.0 kbps
Default: 6.0 kbps (good trade-off between quality and bandwidth for collab)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger("Claudio.NeuralCodec")


@dataclass
class CodecFrame:
    """A single encoded audio frame ready for network transmission."""
    codes: np.ndarray  # (n_codebooks, n_frames) int16
    sample_rate: int = 24_000

    def to_bytes(self) -> bytes:
        """Serialize to compact binary for WebSocket/WebRTC transmission."""
        return self.codes.astype(np.int16).tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, n_codebooks: int) -> CodecFrame:
        """Deserialize from binary."""
        codes = np.frombuffer(data, dtype=np.int16).reshape(n_codebooks, -1)
        return cls(codes=codes)

    @property
    def byte_size(self) -> int:
        return self.codes.size * 2


class NeuralCodec:
    """EnCodec-based audio codec for Claudio's audio transport layer.

    This is the REAL audio path — high-fidelity neural compression
    that produces near-transparent audio at 6-24 kbps.

    The intent pipeline (F0/MFCC/onset) runs in PARALLEL as a metadata
    channel for visualization, coaching, and AI intelligence.
    """

    def __init__(
        self,
        bandwidth_kbps: float = 6.0,
        device: str | None = None,
    ) -> None:
        from encodec import EncodecModel

        self._model = EncodecModel.encodec_model_24khz()
        self._model.set_target_bandwidth(bandwidth_kbps)
        self._model.eval()

        self.sample_rate = self._model.sample_rate  # 24000
        self.channels = self._model.channels  # 1
        self.bandwidth_kbps = bandwidth_kbps

        # Device selection: MPS → CUDA → CPU
        if device:
            self._device = torch.device(device)
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._model.to(self._device)

        # Query actual codebook count from model (don't assume formula)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.sample_rate, device=self._device)
            dummy_enc = self._model.encode(dummy)
            self._n_codebooks = dummy_enc[0][0].shape[1]

        logger.info(
            "NeuralCodec initialized: %s @ %.1f kbps (%d codebooks) on %s",
            "encodec_24khz", bandwidth_kbps, self._n_codebooks, self._device,
        )

    @property
    def n_codebooks(self) -> int:
        """Number of codebooks for current bandwidth setting."""
        return self._n_codebooks

    def encode(self, audio: np.ndarray, input_sr: int = 48_000) -> CodecFrame:
        """Encode audio to compressed codec frame.

        Parameters
        ----------
        audio : np.ndarray, shape (N,), float32, mono
        input_sr : int, sample rate of input audio

        Returns
        -------
        CodecFrame with compressed codes
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio (1D), got shape {audio.shape}")

        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        # Resample if needed
        if input_sr != self.sample_rate:
            from encodec.utils import convert_audio
            audio_t = convert_audio(audio_t, input_sr, self.sample_rate, self.channels)

        audio_t = audio_t.to(self._device)

        with torch.no_grad():
            encoded = self._model.encode(audio_t)

        # Extract codes: encoded is [(codes, scale)]
        codes = encoded[0][0]  # (1, n_codebooks, n_frames)
        return CodecFrame(
            codes=codes.squeeze(0).cpu().numpy(),
            sample_rate=self.sample_rate,
        )

    def decode(self, frame: CodecFrame, target_sr: int = 48_000) -> np.ndarray:
        """Decode a codec frame back to audio.

        Parameters
        ----------
        frame : CodecFrame from encode()
        target_sr : desired output sample rate

        Returns
        -------
        np.ndarray, shape (N,), float32, mono
        """
        codes_t = torch.from_numpy(frame.codes).unsqueeze(0).to(self._device)  # (1, n_cb, n_frames)
        # EnCodec expects [(codes, scale)]
        encoded = [(codes_t, None)]

        with torch.no_grad():
            decoded = self._model.decode(encoded)

        audio = decoded.squeeze().cpu().numpy()

        # Resample to target if needed
        if target_sr != self.sample_rate:
            import torchaudio
            audio_t = torch.from_numpy(audio).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(self.sample_rate, target_sr)
            audio = resampler(audio_t).squeeze().numpy()

        return audio.astype(np.float32)

    def set_bandwidth(self, kbps: float) -> None:
        """Change codec bandwidth at runtime (e.g., adaptive bitrate)."""
        self._model.set_target_bandwidth(kbps)
        self.bandwidth_kbps = kbps
        logger.info("NeuralCodec bandwidth changed to %.1f kbps", kbps)
