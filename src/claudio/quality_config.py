"""
quality_config.py — Shared configuration and helpers for audio quality proofs.

Constants, plot style, signal generators, and processing helpers used by
all quality measurement modules.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from claudio.hrtf_engine import AudioSource, HRTFBinauralEngine
from claudio.realtime_hifi import HiFiProcessor
from claudio.signal_flow_config import SignalFlowConfig

matplotlib.use("Agg")  # Non-interactive backend — no display needed

# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE = 48000
BLOCK_SIZE = 512
OUTPUT_DIR = "demo_output/quality_proof"

# Plot style — dark professional theme
COLORS = {
    "bg": "#0a0a0f",
    "card": "#12121a",
    "grid": "#1e1e2e",
    "text": "#e2e8f0",
    "muted": "#64748b",
    "accent": "#6366f1",
    "green": "#22c55e",
    "red": "#ef4444",
    "cyan": "#22d3ee",
    "amber": "#f59e0b",
    "purple": "#a855f7",
}


def setup_plot_style() -> None:
    """Apply dark professional plot theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["card"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.5,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.grid": True,
            "figure.dpi": 150,
        }
    )


def save_plot(fig, name: str) -> str:
    """Save a figure to the output directory."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", facecolor=COLORS["bg"], pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ─── Signal Generators ───────────────────────────────────────────────────────


def gen_sine(freq: float, duration: float, amp: float = 0.8) -> np.ndarray:
    """Generate a sine wave test signal."""
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def gen_sweep(f_start: float, f_end: float, duration: float, amp: float = 0.8) -> np.ndarray:
    """Generate a logarithmic frequency sweep."""
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE
    phase = (
        2 * np.pi * f_start * duration / np.log(f_end / f_start) * (np.exp(t / duration * np.log(f_end / f_start)) - 1)
    )
    return (np.sin(phase) * amp).astype(np.float32)


def gen_impulse(length: int = 4096) -> np.ndarray:
    """Generate a unit impulse."""
    buf = np.zeros(length, dtype=np.float32)
    buf[0] = 1.0
    return buf


def gen_noise(duration: float, amp: float = 0.3) -> np.ndarray:
    """Generate white noise."""
    n = int(SAMPLE_RATE * duration)
    return (np.random.randn(n) * amp).astype(np.float32)


# ─── Processing Helpers ──────────────────────────────────────────────────────


def process_through_engine(audio: np.ndarray, position=None) -> tuple[np.ndarray, np.ndarray]:
    """Process audio through the HRTF engine."""
    if position is None:
        position = np.array([0.0, 0.0, -2.0])
    cfg = SignalFlowConfig(
        capture_sample_rate=SAMPLE_RATE,
        render_sample_rate=SAMPLE_RATE,
        fft_size=BLOCK_SIZE,
        hrir_length=256,
    )
    engine = HRTFBinauralEngine(config=cfg)
    src = AudioSource(source_id="test", position=position)
    engine.add_source(src)

    n_blocks = len(audio) // BLOCK_SIZE
    out_l, out_r = [], []
    for b in range(n_blocks):
        chunk = audio[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
        frame = engine.render({"test": chunk})
        out_l.append(frame.left)
        out_r.append(frame.right)

    return np.concatenate(out_l), np.concatenate(out_r)


def process_through_hifi(audio: np.ndarray, mode: str = "CLEAN") -> np.ndarray:
    """Process audio through the HiFi processor."""
    proc = HiFiProcessor(SAMPLE_RATE)
    proc._mode = mode
    proc._input_gain = 1.0  # Unity gain for measurement
    proc._output_gain = 1.0
    n_blocks = len(audio) // BLOCK_SIZE
    out = []
    for b in range(n_blocks):
        chunk = audio[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
        stereo = proc.process_block(chunk)
        out.append(stereo[:, 0])
    return np.concatenate(out)
