"""
src/data/audio_dataset.py
==========================
PyTorch Dataset for audio clips.

Loads all WAV files from a directory, resamples to 44 100 Hz,
converts to mono, and returns fixed-length float32 clips.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False


class AudioDataset(Dataset):
    SR = 44_100

    def __init__(self, data_dir: str | Path, clip_seconds: float = 3.0) -> None:
        self.paths       = sorted(Path(data_dir).rglob("*.wav"))
        self.clip_len    = int(clip_seconds * self.SR)
        if not self.paths:
            raise ValueError(f"No WAV files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        if _HAS_TORCHAUDIO:
            waveform, sr = torchaudio.load(str(path))
            if sr != self.SR:
                waveform = torchaudio.functional.resample(waveform, sr, self.SR)
        else:
            import array
            import wave
            with wave.open(str(path)) as wf:
                raw = wf.readframes(wf.getnframes())
                samples = array.array('h', raw)
                waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0

        # Downmix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        audio = waveform.squeeze(0)

        # pad / random-crop to fixed length
        if audio.shape[0] < self.clip_len:
            audio = torch.nn.functional.pad(audio, (0, self.clip_len - audio.shape[0]))
        elif audio.shape[0] > self.clip_len:
            start = random.randint(0, audio.shape[0] - self.clip_len)
            audio = audio[start : start + self.clip_len]

        return audio.unsqueeze(0)   # (1, clip_len) — batch expects (B, C, T) but model wants (B, T)
