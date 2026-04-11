"""
spectrogram_diff.py - Visual Metrology Tool

Generates high-resolution spectrogram diffs between reference and regenerated audio. 
Saves the output to an image for visual verification of transient and frequency gaps.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def generate_spectrogram_diff(ref_path, test_path, out_path, sr=44100):
    # Load audio
    ref_wav, _ = librosa.load(ref_path, sr=sr, mono=True)
    test_wav, _ = librosa.load(test_path, sr=sr, mono=True)
    
    # Align and pad
    max_len = max(len(ref_wav), len(test_wav))
    ref_wav = np.pad(ref_wav, (0, max_len - len(ref_wav)))
    test_wav = np.pad(test_wav, (0, max_len - len(test_wav)))
    
    # Calculate STFT
    hop_length = 512
    D_ref = librosa.stft(ref_wav, hop_length=hop_length)
    D_test = librosa.stft(test_wav, hop_length=hop_length)
    
    # Magnitudes in dB
    S_ref_db = librosa.amplitude_to_db(np.abs(D_ref), ref=np.max)
    S_test_db = librosa.amplitude_to_db(np.abs(D_test), ref=np.max)
    
    # Difference (Absolute error)
    S_diff = np.abs(S_ref_db - S_test_db)
    
    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    img1 = librosa.display.specshow(S_ref_db, x_axis='time', y_axis='hz', sr=sr, ax=ax[0], cmap='magma')
    ax[0].set_title('Reference Audio (Original Stem)')
    fig.colorbar(img1, ax=ax[0], format='%+2.0f dB')
    
    img2 = librosa.display.specshow(S_test_db, x_axis='time', y_axis='hz', sr=sr, ax=ax[1], cmap='magma')
    ax[1].set_title('Claudio Regenerated Audio (Polyphonic Codec)')
    fig.colorbar(img2, ax=ax[1], format='%+2.0f dB')
    
    img3 = librosa.display.specshow(S_diff, x_axis='time', y_axis='hz', sr=sr, ax=ax[2], cmap='inferno')
    ax[2].set_title('Absolute Difference Map (Heat = Lost Information)')
    fig.colorbar(img3, ax=ax[2], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"✅ Spectrogram visual diff saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    generate_spectrogram_diff(args.ref, args.test, args.out)
