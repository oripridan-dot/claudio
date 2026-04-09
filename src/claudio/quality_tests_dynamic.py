"""
quality_tests_dynamic.py — Impulse, Dynamic Range, and Latency Tests

Tests 3, 5, 7 from the audio quality proof suite.
Extracted from audio_quality_proof.py for 300-line compliance.
"""

from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np

from claudio.quality_config import (
    BLOCK_SIZE,
    COLORS,
    SAMPLE_RATE,
    gen_impulse,
    gen_sine,
    process_through_engine,
    process_through_hifi,
    save_plot,
)
from claudio.realtime_hifi import HiFiProcessor


def test_impulse_response() -> dict:
    """Measure impulse response and transient fidelity."""
    print("\n  ── TEST 3: Impulse Response ──")
    impulse = gen_impulse(8192)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    modes = [("CLEAN", COLORS["green"]), ("ENHANCED", COLORS["cyan"]), ("CRYSTAL", COLORS["purple"])]
    results = {}

    for idx, (mode, color) in enumerate(modes):
        out = process_through_hifi(impulse, mode)
        ax = axes.flat[idx]
        time_ms = np.arange(min(512, len(out))) / SAMPLE_RATE * 1000
        ax.plot(time_ms, out[: len(time_ms)], color=color, linewidth=1.0)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{mode} — Impulse Response", fontweight="bold")
        ax.set_xlim(0, time_ms[-1])

        peak = float(np.max(np.abs(out)))
        energy = float(np.sum(out**2))
        results[mode] = {"peak": peak, "energy": energy}
        ax.text(
            0.98,
            0.95,
            f"Peak: {peak:.4f}\nEnergy: {energy:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            color=color,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"facecolor": COLORS["bg"], "alpha": 0.8, "edgecolor": color},
        )

    # HRTF impulse
    out_l, out_r = process_through_engine(impulse)
    ax = axes.flat[3]
    time_ms = np.arange(min(512, len(out_l))) / SAMPLE_RATE * 1000
    ax.plot(time_ms, out_l[: len(time_ms)], color=COLORS["cyan"], linewidth=1.0, alpha=0.8, label="Left")
    ax.plot(time_ms, out_r[: len(time_ms)], color=COLORS["amber"], linewidth=1.0, alpha=0.8, label="Right")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("HRTF Spatial — Impulse Response (L/R)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim(0, time_ms[-1])

    fig.suptitle("Impulse Response Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_plot(fig, "03_impulse_response")

    return results


def test_dynamic_range() -> dict:
    """Measure noise floor with silence input and peak headroom."""
    print("\n  ── TEST 5: Dynamic Range ──")
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    modes = [("CLEAN", COLORS["green"]), ("ENHANCED", COLORS["cyan"]), ("CRYSTAL", COLORS["purple"])]
    results = {}

    for idx, (mode, color) in enumerate(modes):
        out = process_through_hifi(silence, mode)
        n_fft = len(out)
        spectrum = np.abs(np.fft.rfft(out))
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)

        ax = axes[idx]
        ax.plot(freqs, spectrum_db, color=color, linewidth=0.6, alpha=0.9)
        ax.set_xlim(20, 20000)
        ax.set_xscale("log")
        ax.set_ylim(-200, 0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dBFS)")
        ax.set_title(f"{mode} — Noise Floor", fontweight="bold")

        peak_noise = float(np.max(np.abs(out)))
        rms_noise = float(np.sqrt(np.mean(out**2)))
        dr_db = -20 * math.log10(rms_noise + 1e-30)
        results[mode] = {"peak_noise": peak_noise, "rms_noise": rms_noise, "dynamic_range_db": dr_db}
        ax.text(
            0.02,
            0.95,
            f"DR: {dr_db:.0f}dB\nPeak: {peak_noise:.2e}",
            transform=ax.transAxes,
            fontsize=9,
            color=color,
            verticalalignment="top",
            bbox={"facecolor": COLORS["bg"], "alpha": 0.8, "edgecolor": color},
        )

    fig.suptitle("Noise Floor & Dynamic Range (Silence Input)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "05_dynamic_range")

    for mode, data in results.items():
        print(f"    {mode:10s}: DR = {data['dynamic_range_db']:.0f}dB, peak noise = {data['peak_noise']:.2e}")

    return results


def test_latency_histogram() -> dict:
    """Measure and visualize render latency distribution."""
    print("\n  ── TEST 7: Latency Histogram ──")
    audio = gen_sine(1000, 5.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    modes = [("CLEAN", COLORS["green"]), ("ENHANCED", COLORS["cyan"]), ("CRYSTAL", COLORS["purple"])]
    results = {}

    for idx, (mode, color) in enumerate(modes):
        proc = HiFiProcessor(SAMPLE_RATE)
        proc._mode = mode
        proc._input_gain = 1.0
        proc._output_gain = 1.0

        n_blocks = len(audio) // BLOCK_SIZE
        times_us = []
        for b in range(n_blocks):
            chunk = audio[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
            t0 = time.perf_counter()
            proc.process_block(chunk)
            times_us.append((time.perf_counter() - t0) * 1e6)

        import statistics

        mean = statistics.mean(times_us)
        p99 = sorted(times_us)[int(0.99 * len(times_us))]
        mx = max(times_us)
        deadline = (BLOCK_SIZE / SAMPLE_RATE) * 1e6

        ax = axes[idx]
        ax.hist(times_us, bins=80, color=color, alpha=0.8, edgecolor=COLORS["bg"])
        ax.axvline(mean, color=COLORS["text"], linestyle="--", linewidth=1.5, label=f"Mean: {mean:.0f}µs")
        ax.axvline(p99, color=COLORS["amber"], linestyle=":", linewidth=1.5, label=f"P99: {p99:.0f}µs")
        ax.axvline(deadline, color=COLORS["red"], linestyle="-", linewidth=2, label=f"Deadline: {deadline:.0f}µs")
        ax.set_xlabel("Render Time (µs)")
        ax.set_ylabel("Count")
        ax.set_title(f"{mode} — Latency Distribution", fontweight="bold")
        ax.legend(fontsize=8)

        headroom = (1 - mean / deadline) * 100
        results[mode] = {"mean_us": mean, "p99_us": p99, "max_us": mx, "headroom_pct": headroom}
        ax.text(
            0.98,
            0.75,
            f"Headroom: {headroom:.1f}%",
            transform=ax.transAxes,
            fontsize=10,
            color=color,
            horizontalalignment="right",
            bbox={"facecolor": COLORS["bg"], "alpha": 0.8, "edgecolor": color},
        )

    fig.suptitle("Real-Time Render Latency Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "07_latency_histogram")

    for mode, data in results.items():
        print(
            f"    {mode:10s}: mean={data['mean_us']:.0f}µs  P99={data['p99_us']:.0f}µs  "
            f"headroom={data['headroom_pct']:.1f}%"
        )

    return results
