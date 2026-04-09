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
        waveform = None

        if _HAS_TORCHAUDIO:
            try:
                waveform, sr = torchaudio.load(str(path))
                if sr != self.SR:
                    waveform = torchaudio.functional.resample(waveform, sr, self.SR)
            except (ImportError, RuntimeError):
                waveform = None  # Fall through to wave fallback

        if waveform is None:
            import wave

            with wave.open(str(path)) as wf:
                raw = wf.readframes(wf.getnframes())
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                if sw == 2:
                    import numpy as np
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    import array
                    samples = array.array('h', raw)
                    samples = torch.tensor(samples, dtype=torch.float32) / 32768.0
                    samples = samples.numpy()
                import numpy as np
                samples_np = np.asarray(samples, dtype=np.float32)
                if n_ch > 1:
                    samples_np = samples_np.reshape(-1, n_ch).mean(axis=1)
                waveform = torch.from_numpy(samples_np).unsqueeze(0)

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

        return audio   # (clip_len,) — DataLoader stacks to (B, T)
