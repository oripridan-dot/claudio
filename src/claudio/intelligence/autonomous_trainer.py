"""
autonomous_trainer.py — Self-Healing DDSP Autonomous Critic & Trainer

A background daemon that monitors a given directory for incoming raw audio stems.
For each stem:
  1. Compares the original high-fidelity audio against Claudio's internal DDSP intent-regeneration.
  2. If the spectral loss is above the premium threshold, it triggers a fine-tuning epoch.
  3. Once the network is updated, it automatically deletes the raw audio.
"""
from __future__ import annotations

import os
import sys
import time
import math
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import scipy.signal

# Ensure src path is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claudio.forge.model.autoencoder import AudioAutoEncoder
from claudio.forge.loss.spectral_loss import MultiScaleSpectralLoss
# Import the existing forge trainer
try:
    from scripts.train_forge import train as trigger_ddsp_training
except ImportError:
    # If called from different cwd, adjust path
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_forge", str(Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "train_forge.py"))
    train_forge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_forge)
    trigger_ddsp_training = train_forge.train

# Global Configuration
INGEST_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "training_ingest"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "forge_model_best.pt"

# The threshold of Spectral Loss allowed. If > this, trigger training.
MAX_ALLOWED_LOSS = 0.28  # Tuning might be required based on MultiScaleSpectralLoss output


def evaluate_audio_fidelity(filepath: Path) -> float:
    """The Critic Agent: Evaluates the AI's ability to recreate this specific audio."""
    sr = 44100
    try:
        waveform, file_sr = sf.read(str(filepath))
    except Exception as e:
        print(f"[Critic] Failed to load {filepath.name}: {e}")
        return 0.0

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    if file_sr != sr:
        num_samples = int(len(waveform) * float(sr) / file_sr)
        waveform = scipy.signal.resample(waveform, num_samples)

    waveform = waveform.astype(np.float32)

    # Process exactly 3 seconds for quick evaluation
    max_samples = 3 * sr
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    codec = AudioAutoEncoder(latent_dim=128).to(device)
    if MODEL_PATH.exists():
        try:
            ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
            if "autoencoder_state_dict" in ckpt:
                codec.load_state_dict(ckpt["autoencoder_state_dict"])
        except Exception as e:
            print(f"[Critic] Model load error: {e}")
    codec.eval()
    
    chunk_size = sr
    n_chunks = len(waveform) // chunk_size
    decoded_chunks = []
    
    with torch.no_grad():
        for c in range(n_chunks):
            chunk = waveform[c*chunk_size : (c+1)*chunk_size]
            audio_tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
            d_chunk = codec(audio_tensor).cpu().numpy()[0]
            decoded_chunks.append(d_chunk)
        
    if not decoded_chunks:
        return 0.0
        
    regen_audio = np.concatenate(decoded_chunks)
    
    # Calculate Multi-Scale Spectral Loss (STFT L1 Distance)
    loss_fn = MultiScaleSpectralLoss(eps=1e-7)
    with torch.no_grad():
        ref_tensor = torch.from_numpy(waveform[:len(regen_audio)]).unsqueeze(0)
        regen_tensor = torch.from_numpy(regen_audio).unsqueeze(0)
        loss_val = loss_fn(regen_tensor, ref_tensor).item()
        
    return loss_val

def loop():
    print("╔════════════════════════════════════════════════════╗")
    print("║ Claudio Autonomous Critic & Self-Improvement Loop  ║")
    print("╚════════════════════════════════════════════════════╝")
    print(f"Monitoring Directory: {INGEST_DIR}")
    
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    while True:
        try:
            wav_files = list(INGEST_DIR.glob("*.wav")) + list(INGEST_DIR.glob("*.WAV"))
            wav_files = [f for f in wav_files if not f.name.startswith("._")]
            
            if not wav_files:
                time.sleep(3)
                continue
                
            print(f"\n[Trainer] New audio detected: {len(wav_files)} files in ingest queue.")
            
            needs_training = False
            total_loss = 0.0
            for wf in wav_files:
                print(f"[Critic] Measuring fidelity of Polyphonic Codec output against original: {wf.name}...")
                loss = evaluate_audio_fidelity(wf)
                print(f"[Critic] Multi-Scale Spectral Loss: {loss:.4f} (Target: < {MAX_ALLOWED_LOSS})")
                
                if loss > MAX_ALLOWED_LOSS or math.isinf(loss) or math.isnan(loss) or not MODEL_PATH.exists():
                    needs_training = True
                
            if needs_training:
                print("\n[Trainer] ⚠️ Fidelity threshold breached. Spawning localized Polyphonic Fine-Tuning Epoch...")
                
                # Backup old model just in case of catastrophic forgetting
                if MODEL_PATH.exists():
                    backup_path = CHECKPOINT_DIR / f"forge_model_backup_{int(time.time())}.pt"
                    # Keep rolling backups lightweight (delete oldest if > 5)
                    backups = sorted(CHECKPOINT_DIR.glob("forge_model_backup_*.pt"))
                    if len(backups) >= 5:
                        backups[0].unlink()
                    os.rename(str(MODEL_PATH), str(backup_path))
                
                # We use lower epochs for continuous learning so it adapts quickly!
                stats = trigger_ddsp_training(
                    data_dir=str(INGEST_DIR),
                    epochs=60, 
                    lr=2e-4,
                    batch_size=8,
                    clip_seconds=2.0,
                    checkpoint_dir=str(CHECKPOINT_DIR),
                    n_clips_per_file=15
                )
                
                print(f"\n[Trainer] ✅ System Adaptation Complete. Final Loss: {stats['final_loss']:.4f}")
            else:
                print("\n[Critic] ✅ Current Polyphonic Codec capabilities meet premium fidelity standards. Skipping epoch.")
                
            # Cleanup processed files per operator mandate to preserve disk space
            print("[Cleaner] Removing ingested raw audio tracks...")
            for wf in wav_files:
                wf.unlink()
                
            print("[Trainer] Pipeline sleeping... standing by for new multi-track stems.")
            
        except Exception as e:
            print(f"[Trainer Error] An issue occurred in the autonomous loop: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(5)

if __name__ == "__main__":
    loop()
