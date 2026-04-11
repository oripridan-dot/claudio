import sys
import os
import torch
import soundfile as sf
import librosa
from pathlib import Path

# Add src to python path for local execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from claudio.forge.model.autoencoder import AudioAutoEncoder
import audio_metrology

def run_validation():
    sr = 44100
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    codec = AudioAutoEncoder(latent_dim=128).to(device)
    
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "forge_model_best.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    codec.load_state_dict(ckpt["autoencoder_state_dict"])
    codec.eval()

    ref_path = "/Users/oripridan/Downloads/AllHailThePowerOfJesusName_MULTITRACKS_WAV/AllHailThePowerOfJesusName_EGuitar_E.wav"
    wav, _ = librosa.load(ref_path, sr=sr, mono=True)

    chunk_size = sr
    n_chunks = len(wav) // chunk_size
    out = []
    
    with torch.no_grad():
        for c in range(n_chunks):
            chunk = wav[c*chunk_size : (c+1)*chunk_size]
            t = torch.from_numpy(chunk).unsqueeze(0).to(device)
            out.append(codec(t).cpu().numpy()[0])

    import numpy as np
    regen = np.concatenate(out)
    
    out_dir = Path(__file__).resolve().parent.parent / "demo_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "regen_eguitar.wav"
    
    sf.write(str(out_path), regen, sr)

    metrics = audio_metrology.measure_fidelity(ref_path, str(out_path))
    print("METRICS:")
    import json
    print(json.dumps(metrics, indent=2))
    
    return ref_path, str(out_path)

if __name__ == "__main__":
    run_validation()
