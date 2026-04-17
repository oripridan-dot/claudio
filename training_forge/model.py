import torch
import torch.nn as nn


class DDSPDecoder(nn.Module):
    """
    DDSP Neural Decoder with GRU temporal context.

    The original frame-by-frame MLP had no memory across frames — it plateaued
    at spectral loss ~11.5 because it could not learn articulations or note
    transitions that span multiple 250Hz frames.

    This version adds a single bidirectional GRU (hidden=256) before the MLP.
    The GRU sees the full sequence and passes contextual hidden states to the
    output heads, enabling the model to learn temporal dynamics (vibrato onset,
    note attack/release, spectral transitions).

    Architecture:
      Embed(f0)=[16], Embed(loud)=[16], Embed(z)=[32]  →  concat [64]
      GRU(64 → 256, bidirectional=True, layers=2)      →  [512]
      MLP(512 → 512 → 512)                             →  [512]
      Head(harmonics) = Softmax([512→60])
      Head(noise)     = Sigmoid([512→65])
      Head(reverb)    = Sigmoid([512→1])
      Head(f0_residual) = Tanh([512→1]) * 0.05
      Head(voiced)    = Sigmoid([512→1])
    """

    def __init__(self, n_harmonics: int = 60, n_noise: int = 65):
        super().__init__()

        # Input feature embeddings
        self.fc_f0 = nn.Linear(1, 16)
        self.fc_loud = nn.Linear(1, 16)
        self.fc_z = nn.Linear(64, 32)

        # GRU for temporal context — processes full sequence, not frame-by-frame
        # bidirectional=True: hidden=256 per direction → 512 total
        self.gru = nn.GRU(
            input_size=64, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1
        )

        # MLP refiner on top of GRU output
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )

        # DDSP output heads
        self.out_harmonics = nn.Linear(512, n_harmonics)
        self.out_noise = nn.Linear(512, n_noise)
        self.out_reverb = nn.Linear(512, 1)
        self.out_f0_residual = nn.Linear(512, 1)
        self.out_voiced = nn.Linear(512, 1)

    def forward(
        self, f0: torch.Tensor, loudness: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Inputs: (batch, time, features)
        # Normalize to stabilize gradients
        f0_norm = f0 / 2000.0  # Hz → [0,1]
        loudness_norm = torch.clamp(loudness, 0.0, 1.0)
        z_norm = (z + 80.0) / 80.0  # dB range [-80,0] → [0,1]

        h_f0 = self.fc_f0(f0_norm)
        h_loud = self.fc_loud(loudness_norm)
        h_z = self.fc_z(z_norm)

        # Shape: (batch, time, 64)
        x = torch.cat([h_f0, h_loud, h_z], dim=-1)

        # GRU: (batch, time, 64) → (batch, time, 512)
        x, _ = self.gru(x)

        # MLP refinement
        x = self.mlp(x)

        # Synthesizer parameter heads
        harmonics = torch.softmax(self.out_harmonics(x), dim=-1)
        noise = torch.sigmoid(self.out_noise(x))
        reverb_mix = torch.sigmoid(self.out_reverb(x))
        f0_residual = torch.tanh(self.out_f0_residual(x)) * 0.05
        voiced_mask = torch.sigmoid(self.out_voiced(x))

        return harmonics, noise, reverb_mix, f0_residual, voiced_mask
