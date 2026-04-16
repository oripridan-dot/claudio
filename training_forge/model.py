import torch
import torch.nn as nn


class DDSPDecoder(nn.Module):
    """
    DDSP Neural Decoder mirroring the frontend WebNN inference layer.
    Maps Intent Tensor [F0, RMS, MFCC] -> Synth Parameters [Harmonics, Noise].
    """
    def __init__(self, n_harmonics=60, n_noise=65):
        super().__init__()
        # Input features:
        # f0: [batch, time, 1]
        # loudness/rms: [batch, time, 1]
        # mfcc: [batch, time, 13]

        # We embed and scale the inputs
        self.fc_f0 = nn.Linear(1, 16)
        self.fc_loud = nn.Linear(1, 16)
        self.fc_mfcc = nn.Linear(13, 32)

        # Decoder MLP
        self.mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # DDSP Outputs (Additive Harmonics & Filtered Noise Magnitudes)
        self.out_harmonics = nn.Linear(128, n_harmonics)
        self.out_noise = nn.Linear(128, n_noise)

    def forward(self, f0, loudness, mfcc):
        # f0, loudness, mfcc expected shape: (batch, time, features)
        h_f0 = self.fc_f0(f0)
        h_loud = self.fc_loud(loudness)
        h_mfcc = self.fc_mfcc(mfcc)

        # Concatenate encoded intents
        x = torch.cat([h_f0, h_loud, h_mfcc], dim=-1)
        x = self.mlp(x)

        # Synthesizer Amplitudes Control
        # Harmonics sum to 1 (spectral distribution), scaled by loudness later in synth
        harmonics = torch.softmax(self.out_harmonics(x), dim=-1)
        # Noise magnitude envelope
        noise = torch.sigmoid(self.out_noise(x))

        return harmonics, noise
