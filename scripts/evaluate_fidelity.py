import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


def load_and_align(ref_path: str, synth_path: str, target_sr: int = 48000):
    """Loads both audio files, ensures matching sample rates, and trims to the shortest length."""
    ref_wav, sr_ref = torchaudio.load(ref_path)
    synth_wav, sr_synth = torchaudio.load(synth_path)

    # Resample if necessary
    if sr_ref != target_sr:
        ref_wav = torchaudio.functional.resample(ref_wav, sr_ref, target_sr)
    if sr_synth != target_sr:
        synth_wav = torchaudio.functional.resample(synth_wav, sr_synth, target_sr)

    # Convert to mono if stereo
    if ref_wav.shape[0] > 1:
        ref_wav = torch.mean(ref_wav, dim=0, keepdim=True)
    if synth_wav.shape[0] > 1:
        synth_wav = torch.mean(synth_wav, dim=0, keepdim=True)

    # Align lengths perfectly to avoid array mismatch errors
    min_len = min(ref_wav.shape[1], synth_wav.shape[1])
    return ref_wav[:, :min_len], synth_wav[:, :min_len], target_sr


def calculate_lsd(ref_wav: torch.Tensor, synth_wav: torch.Tensor, n_fft: int = 2048) -> float:
    """Calculates Log-Spectral Distance (LSD) - highly sensitive to missing high-frequency textures."""
    spectrogram = T.Spectrogram(n_fft=n_fft, power=2.0)

    ref_spec = spectrogram(ref_wav).squeeze() + 1e-8
    synth_spec = spectrogram(synth_wav).squeeze() + 1e-8

    log_ref = 10 * torch.log10(ref_spec)
    log_synth = 10 * torch.log10(synth_spec)

    # LSD Formula: mean over time of the RMS difference of log spectra over frequency
    lsd = torch.mean(torch.sqrt(torch.mean((log_ref - log_synth) ** 2, dim=0)))
    return lsd.item()


def calculate_mcd(ref_wav: torch.Tensor, synth_wav: torch.Tensor, sr: int) -> float:
    """Calculates Mel-Cepstral Distortion (MCD) - measures timbre/color accuracy."""
    mfcc_transform = T.MFCC(sample_rate=sr, n_mfcc=13, melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 80})

    ref_mfcc = mfcc_transform(ref_wav).squeeze()
    synth_mfcc = mfcc_transform(synth_wav).squeeze()

    # Exclude the 0th coefficient (energy) for pure timbre comparison
    ref_mfcc = ref_mfcc[1:, :]
    synth_mfcc = synth_mfcc[1:, :]

    # MCD Formula: scaled Euclidean distance
    diff = ref_mfcc - synth_mfcc
    mcd = (10.0 * np.sqrt(2.0) / np.log(10.0)) * torch.mean(torch.sqrt(torch.sum(diff**2, dim=0)))
    return mcd.item()


def plot_missing_frequencies(ref_wav: torch.Tensor, synth_wav: torch.Tensor, sr: int, save_path="fidelity_delta.png"):
    """Generates a heatmap showing exactly what frequencies Claudio missed."""
    spectrogram = T.Spectrogram(n_fft=2048, power=1.0)

    # Get magnitude spectrograms
    ref_mag = spectrogram(ref_wav).squeeze()
    synth_mag = spectrogram(synth_wav).squeeze()

    # Calculate the Delta (Difference)
    # Positive values mean the Original WAV had energy that Claudio missed
    # Negative values mean Claudio added artifacts that weren't in the original
    delta_mag = ref_mag - synth_mag

    # Convert to dB for visual clarity
    delta_db = 20 * torch.log10(torch.abs(delta_mag) + 1e-6)

    plt.figure(figsize=(12, 6))

    # Plotting the "Missing" Data
    plt.imshow(
        delta_db.numpy(),
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-20,
        vmax=20,
        extent=[0, ref_wav.shape[1] / sr, 0, sr / 2000],
    )

    plt.colorbar(label="Amplitude Delta (dB) \n Red = Missing in Claudio, Blue = Added Artifacts")
    plt.title("Fidelity Delta: Original WAV vs. Claudio Synthesis")
    plt.ylabel("Frequency (kHz)")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[*] Visual Delta Heatmap saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claudio DDSP Fidelity Diagnostics")
    parser.add_argument("--ref", type=str, required=True, help="Path to ground-truth WAV")
    parser.add_argument("--synth", type=str, required=True, help="Path to Claudio output WAV")
    args = parser.parse_args()

    print("\nAnalyzing Fidelity...")
    print(f"Reference: {args.ref}")
    print(f"Synthesis: {args.synth}\n")

    ref_wav, synth_wav, sr = load_and_align(args.ref, args.synth)

    # 1. Calculate Timbre Loss (MCD)
    mcd_score = calculate_mcd(ref_wav, synth_wav, sr)
    print(f"[!] Mel-Cepstral Distortion (MCD): {mcd_score:.2f}")
    if mcd_score < 4.0:
        print("    -> Excellent timbre match.")
    elif mcd_score < 8.0:
        print("    -> Good, but slight 'plastic' coloration.")
    else:
        print("    -> Poor timbre tracking. Investigate DDSP harmonic mappings.")

    # 2. Calculate High-Frequency / Texture Loss (LSD)
    lsd_score = calculate_lsd(ref_wav, synth_wav)
    print(f"\n[!] Log-Spectral Distance (LSD): {lsd_score:.2f}")
    if lsd_score < 1.0:
        print("    -> SOTA texture rendering. Breaths and friction are perfect.")
    elif lsd_score < 2.5:
        print("    -> Decent, but missing some high-end 'air'.")
    else:
        print("    -> Missing complex textures. Synthesizer sounds robotic/sterile.")

    # 3. Generate Visual Heatmap
    plot_missing_frequencies(ref_wav, synth_wav, sr)
    print("\nDone.")
