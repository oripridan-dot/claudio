import os
import librosa
import soundfile as sf
import warnings

OUT_DIR = "training_forge/data/multitracks"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    warnings.filterwarnings('ignore')
    
    # Librosa provides several high-quality built-in examples that represent diverse multitracks
    examples = {
        'drum_loop': 'vibeace',
        'clean_vocal': 'humpback',       # A challenging spectral target for DDSP
        'brass': 'trumpet',
        'orchestral': 'nutcracker',
        'cello': 'brahms'
    }
    
    print("🥁 Fetching high-quality multitracks...")
    for name, lib_key in examples.items():
        try:
            path = librosa.util.example(lib_key)
            if path:
                y, sr = librosa.load(path, sr=48000)
                dest = os.path.join(OUT_DIR, f"{name}.wav")
                sf.write(dest, y, 48000)
                print(f"  [ok] Downloaded {name} -> 48kHz")
        except Exception as e:
            print(f"  [fail] Failed to download {name}: {e}")
            
    print(f"\n✅ Multitrack fetch complete. Stems stored in {OUT_DIR}/")

if __name__ == '__main__':
    main()
