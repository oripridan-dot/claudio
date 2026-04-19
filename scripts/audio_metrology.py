"""
audio_metrology.py - Autonomous Acoustic "Ears" for the Agent

A command-line script that calculates objective mathematical fidelity metrics
between a reference high-fidelity audio stem and the AI-regenerated version.
Provides JSON output for the AI to dynamically adjust architecture or training.
"""

import argparse
import json

import librosa
import numpy as np
import scipy.signal


def load_and_align(ref_path, test_path, sr=44100):
    """Loads and time-aligns two audio arrays to compensate for latency."""
    ref_wav, _ = librosa.load(ref_path, sr=sr, mono=True)
    test_wav, _ = librosa.load(test_path, sr=sr, mono=True)

    # Pad to same length
    max_len = max(len(ref_wav), len(test_wav))
    ref_wav = np.pad(ref_wav, (0, max_len - len(ref_wav)))
    test_wav = np.pad(test_wav, (0, max_len - len(test_wav)))

    # Align signals using cross-correlation
    corr = scipy.signal.correlate(ref_wav, test_wav, mode="full")
    delay = np.argmax(corr) - (len(test_wav) - 1)

    if delay > 0:
        test_wav = np.pad(test_wav, (delay, 0))[:-delay]
    elif delay < 0:
        test_wav = test_wav[-delay:]
        test_wav = np.pad(test_wav, (0, -delay))

    return ref_wav, test_wav


def calculate_snr(ref, test):
    """Signal-to-Noise Ratio (dB)"""
    noise = ref - test
    signal_power = np.sum(ref**2)
    noise_power = np.sum(noise**2)
    if noise_power == 0:
        return 100.0
    return 10 * np.log10((signal_power / noise_power) + 1e-10)


def calculate_thd_n(ref, test):
    """Total Harmonic Distortion + Noise (%)"""
    noise = ref - test
    rms_noise = np.sqrt(np.mean(noise**2))
    rms_signal = np.sqrt(np.mean(ref**2))
    if rms_signal == 0:
        return 0.0
    return (rms_noise / rms_signal) * 100.0


def calculate_hf_rolloff(test, sr=44100):
    """High Frequency Rolloff (95%) in kHz"""
    rolloff = librosa.feature.spectral_rolloff(y=test, sr=sr, roll_percent=0.95)
    return float(np.mean(rolloff)) / 1000.0


def calculate_phase_coherence(ref, test):
    """Cosine similarity in time domain"""
    dot = np.dot(ref, test)
    norm_r = np.linalg.norm(ref)
    norm_t = np.linalg.norm(test)
    if norm_r == 0 or norm_t == 0:
        return 0.0
    return float(dot / (norm_r * norm_t))


def measure_fidelity(ref_path, test_path, sr=44100):
    ref_wav, test_wav = load_and_align(ref_path, test_path, sr=sr)

    results = {
        "snr_db": round(float(calculate_snr(ref_wav, test_wav)), 2),
        "thd_n_pct": round(float(calculate_thd_n(ref_wav, test_wav)), 2),
        "high_freq_rolloff_khz": round(float(calculate_hf_rolloff(test_wav, sr)), 2),
        "phase_coherence": round(float(calculate_phase_coherence(ref_wav, test_wav)), 4),
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Claudio Audio Quality")
    parser.add_argument("--ref", required=True, help="Original pristine source audio")
    parser.add_argument("--test", required=True, help="Claudio regenerated audio")
    args = parser.parse_args()

    metrics = measure_fidelity(args.ref, args.test)
    print(json.dumps(metrics, indent=2))
