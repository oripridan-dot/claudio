import torch
import torch.nn as nn
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram."""
    x_stft = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        return_complex=True,
        pad_mode='reflect'
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
        loss_log = F.l1_loss(torch.log(torch.clamp(x_mag, min=1e-7)),
                             torch.log(torch.clamp(y_mag, min=1e-7)))
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
