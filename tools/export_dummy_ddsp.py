import torch
import torch.nn as nn
import os

class DummyDDSP(nn.Module):
    def __init__(self, n_harmonics=60, n_noise=65):
        super().__init__()
        # Input features:
        # f0: [batch, time, 1]
        # loudness: [batch, time, 1]
        # z (mfcc/latents): [batch, time, 13]
        
        self.fc_f0 = nn.Linear(1, 16)
        self.fc_loud = nn.Linear(1, 16)
        self.fc_z = nn.Linear(13, 32)
        
        self.hidden = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.out_harmonics = nn.Linear(128, n_harmonics)
        self.out_noise = nn.Linear(128, n_noise)

    def forward(self, f0, loudness, z):
        h_f0 = self.fc_f0(f0)
        h_loud = self.fc_loud(loudness)
        h_z = self.fc_z(z)
        
        x = torch.cat([h_f0, h_loud, h_z], dim=-1)
        x = self.hidden(x)
        
        # Softmax over harmonic amplitudes
        harmonics = torch.softmax(self.out_harmonics(x), dim=-1)
        # Sigmoid over noise magnitudes
        noise = torch.sigmoid(self.out_noise(x))
        
        return harmonics, noise

def main():
    model = DummyDDSP()
    model.eval()
    
    # Dummy inputs: batch=1, time=1
    f0 = torch.randn(1, 1, 1)
    loudness = torch.randn(1, 1, 1)
    z = torch.randn(1, 1, 13)
    
    out_dir = "frontend/public/models"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ddsp_model.onnx")
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        (f0, loudness, z),
        out_path,
        input_names=['f0', 'loudness', 'z'],
        output_names=['harmonics', 'noise'],
        dynamic_axes={
            'f0': {0: 'batch', 1: 'time'},
            'loudness': {0: 'batch', 1: 'time'},
            'z': {0: 'batch', 1: 'time'},
            'harmonics': {0: 'batch', 1: 'time'},
            'noise': {0: 'batch', 1: 'time'},
        },
        opset_version=14
    )
    print(f"Exported dummy DDSP model to {out_path}")

if __name__ == '__main__':
    main()
