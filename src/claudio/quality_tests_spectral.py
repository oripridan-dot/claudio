"""
quality_tests_spectral.py — Frequency Response, Spectrum, and Waterfall Tests

Tests 2, 4, 6 from the audio quality proof suite.
Extracted from audio_quality_proof.py for 300-line compliance.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

from claudio.quality_config import (
    COLORS,
    SAMPLE_RATE,
    gen_sine,
    gen_sweep,
    process_through_engine,
    process_through_hifi,
    save_plot,
)


def test_freq_response() -> dict:
    """Measure frequency response flatness using log sweep."""
    print("\n  ── TEST 2: Frequency Response ──")
    sweep = gen_sweep(20, 20000, 2.0, amp=0.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    modes = [
        ("CLEAN", COLORS["green"]),
        ("ENHANCED", COLORS["cyan"]),
        ("CRYSTAL", COLORS["purple"]),
    ]
    results = {}

    ref_spectrum = np.abs(np.fft.rfft(sweep * np.hanning(len(sweep))))

    for idx, (mode, color) in enumerate(modes):
        out = process_through_hifi(sweep, mode)
        out_spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
        out_freqs = np.fft.rfftfreq(len(out), 1 / SAMPLE_RATE)

        ref_trimmed = ref_spectrum[:len(out_spectrum)]
        transfer = out_spectrum / (ref_trimmed + 1e-10)
        transfer_db = 20 * np.log10(transfer + 1e-10)

        ax = axes.flat[idx]
        ax.semilogx(out_freqs[1:], transfer_db[1:], color=color, linewidth=1.5, alpha=0.9)
        ax.axhline(0, color=COLORS["muted"], linestyle="--", alpha=0.5)
        ax.set_xlim(20, 20000)
        ax.set_ylim(-20, 20)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_title(f"{mode} — Frequency Response", fontweight="bold")

        mask = (out_freqs >= 100) & (out_freqs <= 16000)
        if np.any(mask):
            band_db = transfer_db[mask]
            flatness = float(np.max(band_db) - np.min(band_db))
            results[mode] = flatness
            ax.text(0.02, 0.95, f"Flatness: ±{flatness/2:.1f}dB",
                    transform=ax.transAxes, fontsize=10, color=color,
                    verticalalignment="top",
                    bbox={"facecolor": COLORS["bg"], "alpha": 0.8, "edgecolor": color})

    out_l, out_r = process_through_engine(sweep)
    out_spectrum_l = np.abs(np.fft.rfft(out_l * np.hanning(len(out_l))))
    out_spectrum_r = np.abs(np.fft.rfft(out_r * np.hanning(len(out_r))))
    out_freqs = np.fft.rfftfreq(len(out_l), 1 / SAMPLE_RATE)
    ref_trimmed = ref_spectrum[:len(out_spectrum_l)]
    transfer_l = 20 * np.log10(out_spectrum_l / (ref_trimmed + 1e-10) + 1e-10)
    transfer_r = 20 * np.log10(out_spectrum_r / (ref_trimmed + 1e-10) + 1e-10)

    ax = axes.flat[3]
    ax.semilogx(out_freqs[1:], transfer_l[1:], color=COLORS["cyan"], linewidth=1.2, alpha=0.8, label="Left")
    ax.semilogx(out_freqs[1:], transfer_r[1:], color=COLORS["amber"], linewidth=1.2, alpha=0.8, label="Right")
    ax.set_xlim(20, 20000)
    ax.set_ylim(-40, 20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title("HRTF Spatial — Frequency Response (L/R)", fontweight="bold")
    ax.legend(loc="upper right")

    fig.suptitle("Frequency Response Analysis — All Modes", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_plot(fig, "02_frequency_response")

    for mode, flatness in results.items():
        print(f"    {mode:10s}: ±{flatness/2:.1f}dB (100Hz–16kHz)")

    return results


def test_spectrum_detail() -> dict:
    """Detailed FFT spectrum of 1kHz sine — shows harmonic purity."""
    print("\n  ── TEST 4: Spectrum Analysis (1kHz) ──")
    audio = gen_sine(1000, 1.0, amp=0.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    modes = [
        ("Reference (Dry)", None, COLORS["text"]),
        ("CLEAN", "CLEAN", COLORS["green"]),
        ("ENHANCED", "ENHANCED", COLORS["cyan"]),
        ("CRYSTAL", "CRYSTAL", COLORS["purple"]),
    ]
    results = {}

    for idx, (label, mode, color) in enumerate(modes):
        if mode is None:
            out = audio
        else:
            out = process_through_hifi(audio, mode)

        n_fft = len(out)
        window = np.hanning(n_fft)
        spectrum = np.abs(np.fft.rfft(out * window))
        spectrum_db = 20 * np.log10(spectrum / np.max(spectrum) + 1e-10)
        freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)

        ax = axes.flat[idx]
        ax.plot(freqs, spectrum_db, color=color, linewidth=0.8, alpha=0.9)
        ax.set_xlim(0, 10000)
        ax.set_ylim(-120, 5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"{label} — Spectrum", fontweight="bold")
        ax.axvline(1000, color=COLORS["red"], linestyle="--", alpha=0.3, linewidth=0.8)

        noise_mask = (freqs > 1500) & (freqs < 8000)
        if np.any(noise_mask):
            noise_floor = float(np.median(spectrum_db[noise_mask]))
            results[label] = noise_floor
            ax.text(0.98, 0.05, f"Noise floor: {noise_floor:.1f}dB",
                    transform=ax.transAxes, fontsize=9, color=color,
                    verticalalignment="bottom", horizontalalignment="right",
                    bbox={"facecolor": COLORS["bg"], "alpha": 0.8, "edgecolor": color})

    fig.suptitle("Spectrum Analysis — 1 kHz Sine Test", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_plot(fig, "04_spectrum_1khz")

    for label, nf in results.items():
        print(f"    {label:20s}: noise floor = {nf:.1f}dB")

    return results


def test_waterfall() -> dict:
    """Spectrogram / waterfall analysis of sweep through processing."""
    print("\n  ── TEST 6: Waterfall Spectrogram ──")
    sweep = gen_sweep(20, 20000, 3.0, amp=0.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    modes = [
        ("Reference (Dry)", None),
        ("CLEAN", "CLEAN"),
        ("ENHANCED", "ENHANCED"),
        ("CRYSTAL", "CRYSTAL"),
    ]

    for idx, (label, mode) in enumerate(modes):
        if mode is None:
            out = sweep
        else:
            out = process_through_hifi(sweep, mode)

        ax = axes.flat[idx]
        f_vals, t_vals, sxx = scipy_signal.spectrogram(
            out, fs=SAMPLE_RATE, nperseg=1024, noverlap=768, nfft=2048,
        )
        sxx_db = 10 * np.log10(sxx + 1e-10)
        im = ax.pcolormesh(t_vals, f_vals, sxx_db, shading="gouraud",
                           cmap="magma", vmin=-80, vmax=0)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(20, 20000)
        ax.set_yscale("log")
        plt.colorbar(im, ax=ax, label="dB")

    fig.suptitle("Waterfall Spectrogram — Sweep 20Hz→20kHz", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_plot(fig, "06_waterfall_spectrogram")

    return {}
