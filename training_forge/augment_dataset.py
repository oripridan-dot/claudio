"""
augment_dataset.py — Claudio Training Data Amplifier

Applies 5 deterministic augmentations to each processed .pt clip:
  1. Original (copy)
  2. Pitch shift +1 semitone
  3. Pitch shift -2 semitones
  4. Time stretch x0.92
  5. Time stretch x1.08 + Gaussian noise (sigma=0.002)

Reads from data/processed/, writes to data/augmented/.
Preserves the feature format {f0, loudness, z, audio} expected by train.py.

Run: uv run python augment_dataset.py
"""
import os

import librosa
import numpy as np
import torch

SR = 48000
HOP_LENGTH = 192
N_MELS = 64

IN_DIR = "data/processed"
OUT_DIR = "data/augmented"

AUGMENTATIONS = [
    {"name": "orig",         "pitch_steps": 0,   "time_rate": 1.0,  "noise_sigma": 0.0},
    {"name": "pitch_up1",    "pitch_steps": 1,   "time_rate": 1.0,  "noise_sigma": 0.0},
    {"name": "pitch_dn2",    "pitch_steps": -2,  "time_rate": 1.0,  "noise_sigma": 0.0},
    {"name": "stretch_slow", "pitch_steps": 0,   "time_rate": 0.92, "noise_sigma": 0.0},
    {"name": "stretch_fast", "pitch_steps": 0,   "time_rate": 1.08, "noise_sigma": 0.002},
]


def load_audio_from_pt(pt_path: str) -> np.ndarray:
    """Reconstruct audio waveform from a stored .pt feature dict."""
    data = torch.load(pt_path, map_location="cpu")
    return data["audio"].numpy().astype(np.float32)


def extract_features(y: np.ndarray) -> dict:
    """Re-extract DDSP-aligned features from a waveform."""
    # Pad to clean hop alignment
    pad_len = (HOP_LENGTH - (len(y) % HOP_LENGTH)) % HOP_LENGTH
    y = np.pad(y, (0, pad_len))

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SR,
        frame_length=2048,
        hop_length=HOP_LENGTH,
    )
    f0 = np.nan_to_num(f0)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP_LENGTH)[0]

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP_LENGTH))
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=1.0)

    min_len = min(len(f0), len(rms), log_mel.shape[1])

    return {
        "f0": torch.tensor(f0[:min_len], dtype=torch.float32).unsqueeze(1),
        "loudness": torch.tensor(rms[:min_len], dtype=torch.float32).unsqueeze(1),
        "z": torch.tensor(log_mel[:, :min_len].T, dtype=torch.float32),
        "audio": torch.tensor(y[: min_len * HOP_LENGTH], dtype=torch.float32),
    }


def augment_waveform(y: np.ndarray, aug: dict) -> np.ndarray:
    """Apply a single augmentation config to a waveform."""
    result = y.copy()

    # Pitch shift (semitones)
    if aug["pitch_steps"] != 0:
        result = librosa.effects.pitch_shift(
            result, sr=SR, n_steps=aug["pitch_steps"]
        )

    # Time stretch
    if aug["time_rate"] != 1.0:
        result = librosa.effects.time_stretch(result, rate=aug["time_rate"])

    # Gaussian noise
    if aug["noise_sigma"] > 0.0:
        noise = np.random.normal(0.0, aug["noise_sigma"], len(result)).astype(
            np.float32
        )
        result = result + noise

    # Normalize to prevent clipping
    peak = np.max(np.abs(result)) + 1e-9
    if peak > 1.0:
        result = result / peak

    return result.astype(np.float32)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    pt_files = sorted(
        [os.path.join(IN_DIR, f) for f in os.listdir(IN_DIR) if f.endswith(".pt")]
    )

    if not pt_files:
        print(f"No .pt files found in {IN_DIR}. Run extract_dataset.py first.")
        return

    print(f"Found {len(pt_files)} source clips. Generating {len(pt_files) * len(AUGMENTATIONS)} augmented clips...")
    written = 0

    for pt_path in pt_files:
        stem = os.path.splitext(os.path.basename(pt_path))[0]
        y = load_audio_from_pt(pt_path)

        for aug in AUGMENTATIONS:
            out_name = f"{stem}__{aug['name']}.pt"
            out_path = os.path.join(OUT_DIR, out_name)

            if os.path.exists(out_path):
                print(f"  [skip] {out_name} already exists")
                written += 1
                continue

            try:
                y_aug = augment_waveform(y, aug)
                features = extract_features(y_aug)
                torch.save(features, out_path)
                print(f"  [ok]   {out_name}  ({features['f0'].shape[0]} frames)")
                written += 1
            except Exception as e:
                print(f"  [FAIL] {out_name}: {e}")

    print(f"\n✅ Augmentation complete. {written} clips written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
