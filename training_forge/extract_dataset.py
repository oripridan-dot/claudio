import os
import argparse
import librosa
import numpy as np
import torch

def extract_features(audio_path, sr=48000, hop_length=192):
    """
    Python replica of the WASM Intent Extractor logic.
    Aligns exactly to the 250Hz frame rate expected by the frontend.
    """
    y, _ = librosa.load(audio_path, sr=sr)
    
    # Pad to ensure hop aligns
    pad_len = (hop_length - (len(y) % hop_length)) % hop_length
    y = np.pad(y, (0, pad_len))
    
    # 1. Pitch
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), 
        sr=sr, frame_length=2048, hop_length=hop_length
    )
    f0 = np.nan_to_num(f0)
    
    # 2. Loudness (RMS mapped to dB then normalized)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    
    # 3. MFCC (Timbre)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S**2), n_mfcc=13)
    
    # Align lengths
    min_len = min(len(f0), len(rms), mfcc.shape[1])
    
    features = {
        'f0': torch.tensor(f0[:min_len], dtype=torch.float32).unsqueeze(1),
        'loudness': torch.tensor(rms[:min_len], dtype=torch.float32).unsqueeze(1),
        'mfcc': torch.tensor(mfcc[:, :min_len].T, dtype=torch.float32),
        'audio': torch.tensor(y[:min_len*hop_length], dtype=torch.float32)
    }
    
    return features

def process_directory(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                audio_path = os.path.join(root, f)
                try:
                    features = extract_features(audio_path)
                    out_path = os.path.join(out_dir, f"{count}_{f}.pt")
                    torch.save(features, out_path)
                    print(f"[{count+1}] Extracted: {f}")
                    count += 1
                except Exception as e:
                    print(f"Failed to process {f}: {e}")
    print(f"Extraction complete. {count} clips saved to {out_dir}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw_wavs")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    args = parser.parse_args()
    
    print(f"Extraction tools initialized. Searching {args.data_dir}...")
    process_directory(args.data_dir, args.out_dir)

if __name__ == '__main__':
    main()
