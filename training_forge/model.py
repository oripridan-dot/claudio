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
        # z: [batch, time, 64] (Log-Mel Spectrogram)

        # We embed and scale the inputs
        self.fc_f0 = nn.Linear(1, 16)
        self.fc_loud = nn.Linear(1, 16)
        self.fc_z = nn.Linear(64, 32)

        # Decoder MLP
        self.mlp = nn.Sequential(
            nn.Linear(64, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )

        # DDSP Outputs (Additive Harmonics & Filtered Noise Magnitudes)
        self.out_harmonics = nn.Linear(512, n_harmonics)
        self.out_noise = nn.Linear(512, n_noise)

    def forward(self, f0, loudness, z):
        # f0, loudness, z expected shape: (batch, time, features)
        # Normalize f0 from Hz (0-2000) to [0,1] to stabilize gradients
        f0_norm = f0 / 2000.0
        # Normalize loudness: clip to safe range
        loudness_norm = torch.clamp(loudness, 0.0, 1.0)
        # Normalize z (log-mel) from [-80,0] dB to [0,1]
        z_norm = (z + 80.0) / 80.0

        h_f0 = self.fc_f0(f0_norm)
        h_loud = self.fc_loud(loudness_norm)
        h_z = self.fc_z(z_norm)

        # Concatenate encoded intents
        x = torch.cat([h_f0, h_loud, h_z], dim=-1)
        x = self.mlp(x)

        # Synthesizer Amplitudes Control
        # Harmonics sum to 1 (spectral distribution), scaled by loudness later in synth
        harmonics = torch.softmax(self.out_harmonics(x), dim=-1)
        # Noise magnitude envelope
        noise = torch.sigmoid(self.out_noise(x))

        return harmonics, noise
