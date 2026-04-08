"""
src/model/feature_extractor.py
================================
Extracts F0 (fundamental frequency) and loudness contours from raw audio
for use as conditioning signals in the DDSP decoder.

Both features are computed at a 4 ms frame rate (250 frames/second at 44100 Hz)
and normalised to [0, 1] before feeding into the GRU encoder.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    Lightweight CPU-friendly F0 + loudness extractor.

    F0 estimation: YIN algorithm (de Cheveigné & Kawahara, 2002).
    Loudness:      RMS energy in A-weighting approximation.
    """

    FRAME_RATE_HZ = 250   # 4 ms frames
    F0_MIN_HZ     = 32.7  # C1
    F0_MAX_HZ     = 2093  # C7

    def __init__(self, sample_rate: int = 44_100) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.hop = sample_rate // self.FRAME_RATE_HZ
        self.frame_len = self.hop * 4   # 4× hop for YIN window

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        audio : (B, T) float32 mono audio, normalised to [-1, 1].

        Returns
        -------
        f0        : (B, N_frames) in [0, 1]  (0 = unvoiced / below min)
        loudness  : (B, N_frames) in [0, 1]
        """
        B, T = audio.shape
        n_frames = (T - self.frame_len) // self.hop + 1
        f0        = torch.zeros(B, n_frames, device=audio.device)
        loudness  = torch.zeros(B, n_frames, device=audio.device)

        for b in range(B):
            frames = audio[b].unfold(0, self.frame_len, self.hop)[:n_frames]
            # Loudness = RMS per frame
            rms = frames.pow(2).mean(-1).sqrt()
            loudness[b] = torch.clamp(rms / 0.3, 0.0, 1.0)  # normalise

            # YIN difference function
            for i, frame in enumerate(frames):
                f  = frame.cpu().numpy()
                f0_hz = self._yin(f)
                if f0_hz and self.F0_MIN_HZ <= f0_hz <= self.F0_MAX_HZ:
                    # Log-normalise F0 to [0, 1]
                    lo = np.log2(self.F0_MIN_HZ)
                    hi = np.log2(self.F0_MAX_HZ)
                    f0[b, i] = (np.log2(f0_hz) - lo) / (hi - lo)

        return f0, loudness

    @staticmethod
    def _yin(frame: np.ndarray, threshold: float = 0.1) -> float | None:
        """YIN F0 estimator.  Returns Hz or None if unvoiced."""
        n = len(frame)
        half = n // 2
        diff = np.zeros(half)
        for tau in range(1, half):
            diff[tau] = np.sum((frame[:half] - frame[tau : tau + half]) ** 2)
        if diff[1:].sum() == 0:
            return None
        # Cumulative mean normalised difference
        cmnd = np.zeros(half)
        cumsum = 0.0
        for tau in range(1, half):
            cumsum += diff[tau]
            cmnd[tau] = diff[tau] * tau / cumsum if cumsum > 0 else 1.0
        # Find first dip below threshold
        for tau in range(2, half - 1):
            if cmnd[tau] < threshold and cmnd[tau] < cmnd[tau + 1]:
                return None if tau == 0 else 44_100 / tau
        return None
