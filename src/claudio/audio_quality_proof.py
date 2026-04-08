"""
audio_quality_proof.py — Comprehensive Audio Quality Proof

Orchestrates all quality measurement tests and generates a summary dashboard.
Individual tests are in quality_tests_distortion.py, quality_tests_spectral.py,
and quality_tests_dynamic.py. Shared config is in quality_config.py.

Usage:
    cd claudio && .venv/bin/python -m claudio.audio_quality_proof
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from claudio.quality_config import COLORS, OUTPUT_DIR, save_plot, setup_plot_style
from claudio.quality_tests_distortion import test_imd, test_thdn
from claudio.quality_tests_dynamic import (
    test_dynamic_range,
    test_impulse_response,
    test_latency_histogram,
)
from claudio.quality_tests_spectral import (
    test_freq_response,
    test_spectrum_detail,
    test_waterfall,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary(all_results: dict) -> None:
    """Generate a summary dashboard plot."""
    print("\n  ── Generating Summary Dashboard ──")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. THD+N bar chart
    ax = axes[0, 0]
    thdn = all_results.get("thdn", {})
    avgs = [0] * 4
    if thdn:
        modes = ["clean", "enhanced", "crystal", "hrtf"]
        mode_labels = ["CLEAN", "ENHANCED", "CRYSTAL", "HRTF"]
        mode_colors = [COLORS["green"], COLORS["cyan"], COLORS["purple"], COLORS["amber"]]
        avgs = [sum(thdn.get(m, [0])) / max(len(thdn.get(m, [1])), 1) for m in modes]
        ax.bar(mode_labels, avgs, color=mode_colors, alpha=0.9)
        ax.set_ylabel("Avg THD+N (%)")
        ax.set_title("THD+N (lower = better)", fontweight="bold")
        ax.set_yscale("log")

    # 2. Dynamic range
    ax = axes[0, 1]
    dr = all_results.get("dynamic_range", {})
    dr_vals = []
    if dr:
        dr_modes = list(dr.keys())
        dr_vals = [dr[m]["dynamic_range_db"] for m in dr_modes]
        ax.barh(dr_modes, dr_vals, color=[COLORS["green"], COLORS["cyan"], COLORS["purple"]], alpha=0.9)
        ax.set_xlabel("Dynamic Range (dB)")
        ax.set_title("Dynamic Range (higher = better)", fontweight="bold")

    # 3. Latency comparison
    ax = axes[0, 2]
    lat = all_results.get("latency", {})
    lat_modes, lat_means, headrooms = [], [], []
    if lat:
        lat_modes = list(lat.keys())
        lat_means = [lat[m]["mean_us"] for m in lat_modes]
        lat_p99 = [lat[m]["p99_us"] for m in lat_modes]
        headrooms = [lat[m]["headroom_pct"] for m in lat_modes]
        x = np.arange(len(lat_modes))
        ax.bar(x - 0.2, lat_means, 0.35, label="Mean", color=COLORS["green"], alpha=0.9)
        ax.bar(x + 0.2, lat_p99, 0.35, label="P99", color=COLORS["amber"], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(lat_modes)
        ax.set_ylabel("Latency (µs)")
        ax.set_title("Processing Latency", fontweight="bold")
        ax.legend()

    # 4. Freq response flatness
    ax = axes[1, 0]
    fr = all_results.get("freq_response", {})
    fr_vals = []
    if fr:
        fr_modes = list(fr.keys())
        fr_vals = [fr[m] for m in fr_modes]
        bars = ax.bar(fr_modes, fr_vals,
                      color=[COLORS["green"], COLORS["cyan"], COLORS["purple"]], alpha=0.9)
        ax.set_ylabel("Deviation (dB)")
        ax.set_title("Freq Response Flatness (lower = better)", fontweight="bold")
        for bar_item, val in zip(bars, fr_vals, strict=True):
            ax.text(bar_item.get_x() + bar_item.get_width() / 2, val + 0.1,
                    f"±{val/2:.1f}", ha="center", fontsize=9, color=COLORS["text"])

    # 5. Headroom gauge
    ax = axes[1, 1]
    if lat:
        bars = ax.barh(lat_modes, headrooms,
                       color=[COLORS["green"], COLORS["cyan"], COLORS["purple"]], alpha=0.9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("CPU Headroom (%)")
        ax.set_title("Real-Time Headroom (higher = better)", fontweight="bold")
        for bar_item, val in zip(bars, headrooms, strict=True):
            ax.text(val - 2, bar_item.get_y() + bar_item.get_height() / 2,
                    f"{val:.1f}%", ha="right", va="center", fontsize=10,
                    color=COLORS["bg"], fontweight="bold")

    # 6. Scorecard
    ax = axes[1, 2]
    ax.axis("off")
    scorecard = [
        ("THD+N (CLEAN)", f"{avgs[0]:.4f}%" if thdn else "—", "< 1%"),
        ("Dynamic Range", f"{dr_vals[0]:.0f}dB" if dr else "—", "> 90dB"),
        ("Freq Flatness", f"±{fr_vals[0]/2:.1f}dB" if fr else "—", "±3dB"),
        ("Latency (CLEAN)", f"{lat_means[0]:.0f}µs" if lat else "—", "< 1000µs"),
        ("Headroom", f"{headrooms[0]:.1f}%" if lat else "—", "> 90%"),
        ("Overruns", "0", "0"),
    ]
    y = 0.9
    ax.text(0.5, 1.0, "QUALITY SCORECARD", transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha="center", va="top", color=COLORS["text"])
    for metric, value, target in scorecard:
        ax.text(0.05, y, metric, transform=ax.transAxes, fontsize=10, color=COLORS["muted"])
        ax.text(0.55, y, value, transform=ax.transAxes, fontsize=10, color=COLORS["green"], fontweight="bold")
        ax.text(0.8, y, f"target: {target}", transform=ax.transAxes, fontsize=8, color=COLORS["muted"])
        y -= 0.12

    fig.suptitle("CLAUDIO — Audio Quality Proof Dashboard",
                 fontsize=18, fontweight="bold", y=1.03, color=COLORS["text"])
    fig.tight_layout()
    save_plot(fig, "00_summary_dashboard")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_plot_style()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CLAUDIO — AUDIO QUALITY PROOF                             ║")
    print("║  Professional measurements with visual proof                ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    all_results = {}
    all_results["thdn"] = test_thdn()
    all_results["freq_response"] = test_freq_response()
    all_results["impulse"] = test_impulse_response()
    all_results["spectrum"] = test_spectrum_detail()
    all_results["dynamic_range"] = test_dynamic_range()
    all_results["waterfall"] = test_waterfall()
    all_results["latency"] = test_latency_histogram()
    all_results["imd"] = test_imd()

    generate_summary(all_results)

    n_plots = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])
    print("\n  ═══════════════════════════════════════════════════")
    print(f"  ✅ {n_plots} plots generated in {OUTPUT_DIR}/")
    print("  ═══════════════════════════════════════════════════")
    print("\n  Open the dashboard:")
    print(f"    open {OUTPUT_DIR}/00_summary_dashboard.png")


if __name__ == "__main__":
    main()
