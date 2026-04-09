"""
quality_tests_distortion.py — THD+N and IMD Measurement Tests

Tests 1 and 8 from the audio quality proof suite.
Extracted from audio_quality_proof.py for 300-line compliance.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from claudio.quality_config import (
    COLORS,
    SAMPLE_RATE,
    gen_sine,
    process_through_engine,
    process_through_hifi,
    save_plot,
)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: THD+N Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def test_thdn() -> dict:
    """Measure THD+N at multiple frequencies."""
    print("\n  ── TEST 1: THD+N Analysis ──")
    freqs = [100, 250, 500, 1000, 2000, 4000, 8000, 16000]
    results = {"freqs": freqs, "clean": [], "enhanced": [], "crystal": [], "hrtf": []}

    for f in freqs:
        audio = gen_sine(f, 0.5)

        for mode_name, mode_key in [("clean", "CLEAN"), ("enhanced", "ENHANCED"), ("crystal", "CRYSTAL")]:
            out = process_through_hifi(audio, mode_key)
            n_fft = len(out)
            spectrum = np.abs(np.fft.rfft(out * np.hanning(n_fft)))
            freqs_axis = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)

            fund_idx = np.argmin(np.abs(freqs_axis - f))
            fund_power = spectrum[fund_idx] ** 2
            total_power = np.sum(spectrum**2)
            noise_power = total_power - fund_power
            thdn_pct = math.sqrt(noise_power / (total_power + 1e-30)) * 100
            results[mode_name].append(thdn_pct)

        # HRTF engine
        out_l, _ = process_through_engine(audio)
        n_fft = len(out_l)
        spectrum = np.abs(np.fft.rfft(out_l * np.hanning(n_fft)))
        freqs_axis = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)
        fund_idx = np.argmin(np.abs(freqs_axis - f))
        fund_power = spectrum[fund_idx] ** 2
        total_power = np.sum(spectrum**2)
        noise_power = total_power - fund_power
        thdn_pct = math.sqrt(noise_power / (total_power + 1e-30)) * 100
        results["hrtf"].append(thdn_pct)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(freqs))
    w = 0.2
    ax.bar(x - 1.5 * w, results["clean"], w, label="CLEAN", color=COLORS["green"], alpha=0.9)
    ax.bar(x - 0.5 * w, results["enhanced"], w, label="ENHANCED", color=COLORS["cyan"], alpha=0.9)
    ax.bar(x + 0.5 * w, results["crystal"], w, label="CRYSTAL", color=COLORS["purple"], alpha=0.9)
    ax.bar(x + 1.5 * w, results["hrtf"], w, label="HRTF Spatial", color=COLORS["amber"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in freqs], rotation=30)
    ax.set_ylabel("THD+N (%)")
    ax.set_title("THD+N vs Frequency — All Processing Modes", fontweight="bold", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.set_ylim(0.001, 100)
    save_plot(fig, "01_thdn_analysis")

    for mode_name in ["clean", "enhanced", "crystal", "hrtf"]:
        avg = sum(results[mode_name]) / len(results[mode_name])
        print(f"    {mode_name.upper():10s}: avg THD+N = {avg:.4f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: IMD — Intermodulation Distortion
# ═══════════════════════════════════════════════════════════════════════════════


def test_imd() -> dict:
    """SMPTE IMD test: 60Hz + 7kHz mixed 4:1."""
    print("\n  ── TEST 8: Intermodulation Distortion (SMPTE) ──")
    duration = 1.0
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    imd_signal = (np.sin(2 * np.pi * 60 * t) * 0.4 + np.sin(2 * np.pi * 7000 * t) * 0.1).astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    modes = [
        ("Reference", None, COLORS["text"]),
        ("CLEAN", "CLEAN", COLORS["green"]),
        ("ENHANCED", "ENHANCED", COLORS["cyan"]),
        ("CRYSTAL", "CRYSTAL", COLORS["purple"]),
    ]
    results = {}

    for idx, (label, mode, color) in enumerate(modes):
        if mode is None:
            out = imd_signal
        else:
            out = process_through_hifi(imd_signal, mode)

        n_fft = len(out)
        spectrum = np.abs(np.fft.rfft(out * np.hanning(n_fft)))
        spectrum_db = 20 * np.log10(spectrum / np.max(spectrum) + 1e-10)
        freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)

        ax = axes.flat[idx]
        ax.plot(freqs, spectrum_db, color=color, linewidth=0.7, alpha=0.9)
        ax.set_xlim(0, 15000)
        ax.set_ylim(-120, 5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"{label} — IMD Spectrum", fontweight="bold")

        ax.axvline(60, color=COLORS["red"], alpha=0.3, linewidth=0.8)
        ax.axvline(7000, color=COLORS["red"], alpha=0.3, linewidth=0.8)
        for imd_freq in [6940, 7060, 6880, 7120]:
            ax.axvline(imd_freq, color=COLORS["amber"], alpha=0.2, linewidth=0.5)

        f7k_idx = np.argmin(np.abs(freqs - 7000))
        imd_indices = [np.argmin(np.abs(freqs - f)) for f in [6940, 7060, 6880, 7120]]
        imd_power = sum(spectrum[i] ** 2 for i in imd_indices)
        fund_power = spectrum[f7k_idx] ** 2
        imd_pct = math.sqrt(imd_power / (fund_power + 1e-30)) * 100
        results[label] = imd_pct

    fig.suptitle("Intermodulation Distortion (SMPTE: 60Hz + 7kHz)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_plot(fig, "08_imd_analysis")

    for label, imd in results.items():
        print(f"    {label:15s}: IMD = {imd:.4f}%")

    return results
