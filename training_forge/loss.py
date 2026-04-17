import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram."""
    x_stft = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        return_complex=True,
        pad_mode="reflect",
    )
    # L1 magnitude safely
    mag = torch.sqrt(torch.clamp(x_stft.real**2 + x_stft.imag**2, min=1e-7))
    return mag


class SpectralLoss(nn.Module):
    def __init__(self, fft_size=1024, hop_size=256, win_length=1024):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x_mag, y_mag):
        # L1 linear loss
        loss_lin = F.l1_loss(x_mag, y_mag)
        # L1 log loss
        loss_log = F.l1_loss(torch.log(torch.clamp(x_mag, min=1e-7)), torch.log(torch.clamp(y_mag, min=1e-7)))
        return loss_lin + loss_log


class MultiScaleSpectralLoss(nn.Module):
    """
    Computes spectral loss across multiple STFT resolutions.
    """

    def __init__(self, fft_sizes=None, hop_sizes=None, win_lengths=None):
        if win_lengths is None:
            win_lengths = [2048, 1024, 512, 256, 128, 64]
        if hop_sizes is None:
            hop_sizes = [512, 256, 128, 64, 32, 16]
        if fft_sizes is None:
            fft_sizes = [2048, 1024, 512, 256, 128, 64]
        super().__init__()
        self.spectral_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths, strict=False):
            self.spectral_losses.append(SpectralLoss(fs, ss, wl))

    def forward(self, x, y):
        # x, y: [batch_size, time]
        loss = 0.0
        for f in self.spectral_losses:
            x_mag = stft(x, f.fft_size, f.hop_size, f.win_length, f.window)
            y_mag = stft(y, f.fft_size, f.hop_size, f.win_length, f.window)
            loss += f(x_mag, y_mag)
        return loss


class MelSpectralLoss(nn.Module):
    """
    Perceptual loss on log-mel spectrogram.
    Penalises errors in the frequency regions the ear is most sensitive to.
    4 STFT scales × 80 mel bands to capture both transient and tonal structure.
    """

    _CONFIGS = [
        # n_mels must be < n_fft//2 + 1 (number of freq bins); keep 10% margin
        {"n_fft": 2048, "hop": 512, "n_mels": 80},  # 1025 bins — 80 safe
        {"n_fft": 1024, "hop": 256, "n_mels": 64},  # 513 bins  — 64 safe
        {"n_fft": 512, "hop": 128, "n_mels": 32},  # 257 bins  — 32 safe
        {"n_fft": 256, "hop": 64, "n_mels": 16},  # 129 bins  — 16 safe
    ]

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.mel_transforms = nn.ModuleList()
        for cfg in self._CONFIGS:
            self.mel_transforms.append(
                T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=cfg["n_fft"],
                    hop_length=cfg["hop"],
                    n_mels=cfg["n_mels"],
                    power=2.0,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: [batch, time]
        loss = torch.zeros(1, device=x.device)
        for mel_tf in self.mel_transforms:
            mel_tf = mel_tf.to(x.device)
            x_mel = torch.log(mel_tf(x) + 1e-7)
            y_mel = torch.log(mel_tf(y) + 1e-7)
            loss = loss + F.l1_loss(x_mel, y_mel)
        return loss


class CombinedPerceptualLoss(nn.Module):
    """
    70% multi-scale spectral L1 + 30% perceptual mel loss.
    Drop-in replacement for MultiScaleSpectralLoss in refine_loop.py.
    """

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.spectral = MultiScaleSpectralLoss()
        self.mel = MelSpectralLoss(sample_rate=sample_rate)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.9 * self.spectral(x, y) + 0.1 * self.mel(x, y)
