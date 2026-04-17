import glob
import json
import os

import librosa
import numpy as np


def calculate_lsd(y_true, y_pred, sr, n_fft=2048, hop_length=512):
    """Log Spectral Distance (LSD)"""
    S_true = np.abs(librosa.stft(y_true, n_fft=n_fft, hop_length=hop_length))
    S_pred = np.abs(librosa.stft(y_pred, n_fft=n_fft, hop_length=hop_length))

    # Avoid log of 0
    S_true = np.maximum(S_true, 1e-10)
    S_pred = np.maximum(S_pred, 1e-10)

    # Log spectra in DB
    log_true = 10.0 * np.log10(S_true**2)
    log_pred = 10.0 * np.log10(S_pred**2)

    # LSD per frame: sqrt( 1/K * sum (log_true - log_pred)^2 )
    diff = log_true - log_pred
    lsd_per_frame = np.sqrt(np.mean(diff**2, axis=0))
    return float(np.mean(lsd_per_frame))


def calculate_mcd(y_true, y_pred, sr):
    """Mel-Cepstral Distortion (MCD)"""
    mfcc_true = librosa.feature.mfcc(y=y_true, sr=sr, n_mfcc=13)
    mfcc_pred = librosa.feature.mfcc(y=y_pred, sr=sr, n_mfcc=13)

    # Align shapes just in case
    min_len = min(mfcc_true.shape[1], mfcc_pred.shape[1])
    diff = mfcc_true[:, :min_len] - mfcc_pred[:, :min_len]

    # MCD formula: (10 / ln(10)) * sqrt(2 * sum(...) )
    distance_per_frame = np.sqrt(np.sum(diff**2, axis=0))
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(distance_per_frame)
    return float(mcd)


def calculate_f0_rmse(y_true, y_pred, sr):
    """F0 RMSE"""
    f0_true, _, _ = librosa.pyin(y_true, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_pred, _, _ = librosa.pyin(y_pred, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)

    # Filter out unvoiced frames (NaNs)
    valid_mask = ~np.isnan(f0_true) & ~np.isnan(f0_pred)
    if not np.any(valid_mask):
        return 0.0

    f0_t = f0_true[valid_mask]
    f0_p = f0_pred[valid_mask]

    rmse = np.sqrt(np.mean((f0_t - f0_p) ** 2))
    return float(rmse)


def calculate_crest_factor(y_true, y_pred):
    """Crest factor (Peak / RMS) indicating transient preservation"""
    cf_t = np.max(np.abs(y_true)) / (np.sqrt(np.mean(y_true**2)) + 1e-10)
    cf_p = np.max(np.abs(y_pred)) / (np.sqrt(np.mean(y_pred**2)) + 1e-10)
    return float(abs(cf_t - cf_p))


def calculate_snr(orig_path: str, gen_path: str, sr: int) -> float:
    """
    Signal-to-Noise Ratio in dB.
    SNR = 10 * log10(signal_power / noise_power)
    where noise = (generated - original).
    Higher is better. Negative SNR means the output is mostly noise.
    Returns 0.0 if either file is missing.
    """
    if not os.path.exists(orig_path) or not os.path.exists(gen_path):
        return 0.0
    y_t, _ = librosa.load(orig_path, sr=sr)
    y_p, _ = librosa.load(gen_path, sr=sr)
    min_len = min(len(y_t), len(y_p))
    y_t = y_t[:min_len]
    y_p = y_p[:min_len]
    signal_power = np.mean(y_t**2) + 1e-10
    noise_power = np.mean((y_t - y_p) ** 2) + 1e-10
    return float(10.0 * np.log10(signal_power / noise_power))


def run_analysis(demo_dir="demo_output"):
    original_files = sorted(glob.glob(os.path.join(demo_dir, "*_original.wav")))
    metrics = {"LSD": [], "MCD": [], "F0_RMSE": [], "CrestFactor_Diff": []}

    if not original_files:
        print(f"No original files found in {demo_dir}")
        return

    for orig_path in original_files:
        gen_path = orig_path.replace("_original.wav", "_generated.wav")
        if not os.path.exists(gen_path):
            continue

        print(f"Analyzing {orig_path} vs {gen_path}")
        # Note: loading at 48kHz for DDSP
        y_t, sr = librosa.load(orig_path, sr=48000)
        y_p, _ = librosa.load(gen_path, sr=sr)

        # Trim to match exactly
        min_len = min(len(y_t), len(y_p))
        y_t = y_t[:min_len]
        y_p = y_p[:min_len]

        metrics["LSD"].append(calculate_lsd(y_t, y_p, sr))
        metrics["MCD"].append(calculate_mcd(y_t, y_p, sr))
        metrics["F0_RMSE"].append(calculate_f0_rmse(y_t, y_p, sr))
        metrics["CrestFactor_Diff"].append(calculate_crest_factor(y_t, y_p))

    final_metrics = {k: float(np.mean(v)) for k, v in metrics.items() if v}
    print("-" * 40)
    print("Aggregate Fidelity Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    with open("fidelity_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    return final_metrics


if __name__ == "__main__":
    run_analysis()
