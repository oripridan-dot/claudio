import os
import warnings

import librosa
import soundfile as sf


def main():
    out_dir = "data/raw_wavs"
    os.makedirs(out_dir, exist_ok=True)

    print("Fetching high-quality open-source reference stems via Librosa...")

    warnings.filterwarnings('ignore')

    stems = {
        'trumpet': librosa.util.example('trumpet')
    }

    for name, path in stems.items():
        if os.path.exists(path):
            y, sr = librosa.load(path, sr=48000)
            dest = os.path.join(out_dir, f"{name}.wav")
            # Save using soundfile (which is already installed alongside librosa)
            sf.write(dest, y, 48000)
            print(f"✅ Downloaded and formatted: {name} (48kHz)")
        else:
            print(f"❌ Failed to resolve {name}.")

    print(f"\nAll files saved to {os.path.abspath(out_dir)}. Ready for feature extraction!")

if __name__ == '__main__':
    main()
