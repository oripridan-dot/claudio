"""
realtime_hifi.py — Claudio Hi-Fi Real-Time Audio POC

Proves that Claudio can process audio in real-time at studio-grade
quality through Bluetooth headphones. No HRTF, no spatial — pure
clean signal processing.

BT Latency Strategy:
  Bluetooth A2DP/AAC adds ~100-150ms of codec/transport latency.
  This is hardware-locked and cannot be reduced. What we CAN prove:
  1. Claudio adds < 1ms of processing latency on top of BT
  2. The audio quality through the pipeline is pristine
  3. Zero buffer overruns = glitch-free streaming

  For pre-recorded content, we pre-fill the output buffer to keep
  the BT codec continuously fed, eliminating BT-side starvation jitter.

Modes:
  CLEAN     — Zero-processing passthrough (baseline quality reference)
  ENHANCED  — Subtle dynamics + presence boost
  CRYSTAL   — Full hi-fi chain: dynamics + air EQ + stereo widening

Usage:
    Terminal.app:  cd ~/claudio && .venv/bin/python -m claudio.realtime_hifi
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

# Smallest block that BT A2DP can handle without starvation
# 256 samples @ 44.1kHz = 5.8ms per block
BLOCK_SIZE = 256


class HiFiProcessor:
    """Clean, high-fidelity audio processor — no HRTF, no spatial."""

    def __init__(self, sample_rate: int) -> None:
        self._sr = sample_rate
        self._mode = "CLEAN"  # CLEAN / ENHANCED / CRYSTAL
        self._input_gain = 6.0  # Mic boost
        self._output_gain = 4.0  # Output makeup
        self._mix = 1.0  # Dry/wet mix (1.0 = full wet)

        # Stats
        self._render_times: list[float] = []
        self._peak_in = 0.0
        self._peak_out = 0.0
        self._block_count = 0
        self._overruns = 0
        self._lock = threading.Lock()

        # Compressor state
        self._comp_env = 0.0

        # Pre-fill buffer for BT jitter compensation
        # We send a few blocks of silence ahead to keep BT codec fed
        self._prefill_blocks = 3
        self._prefilled = False

    def process_block(self, mono_in: np.ndarray) -> np.ndarray:
        """Process a single block of mono audio → stereo output."""
        t0 = time.perf_counter()

        # Input gain
        signal = mono_in * self._input_gain

        # Track input peak
        in_peak = float(np.max(np.abs(signal)))
        self._peak_in = max(self._peak_in, in_peak)

        if self._mode == "CLEAN":
            left = signal
            right = signal
        elif self._mode == "ENHANCED":
            left, right = self._enhanced(signal)
        else:  # CRYSTAL
            left, right = self._crystal(signal)

        # Output gain with soft limiter
        left = self._soft_limit(left * self._output_gain)
        right = self._soft_limit(right * self._output_gain)

        elapsed_us = (time.perf_counter() - t0) * 1e6

        # Track stats
        out_peak = float(max(np.max(np.abs(left)), np.max(np.abs(right))))
        self._peak_out = max(self._peak_out, out_peak)

        with self._lock:
            self._render_times.append(elapsed_us)
            if len(self._render_times) > 1000:
                self._render_times = self._render_times[-500:]
        self._block_count += 1

        # Interleave to stereo
        stereo = np.empty((len(left), 2), dtype=np.float32)
        stereo[:, 0] = left
        stereo[:, 1] = right
        return stereo

    # ── Processing Modes ──────────────────────────────────────────────

    def _enhanced(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Subtle dynamics + presence. Transparent, clean boost."""
        # Gentle compression (2:1 ratio, -20dB threshold)
        compressed = self._compress(signal, threshold=0.1, ratio=2.0, attack_ms=5.0, release_ms=50.0)
        # Presence boost: gentle high-shelf via simple difference
        # Adds clarity without harshness
        boosted = self._presence_boost(compressed, amount=0.3)
        return boosted, boosted

    def _crystal(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full hi-fi chain: dynamics + air + stereo width."""
        # Compression for consistent dynamics
        compressed = self._compress(signal, threshold=0.08, ratio=3.0, attack_ms=2.0, release_ms=30.0)
        # Presence + air
        bright = self._presence_boost(compressed, amount=0.5)
        # Subtle stereo widening via Haas effect (tiny delay on one channel)
        delay_samples = max(1, int(0.0003 * self._sr))  # 0.3ms
        left = bright
        right = np.roll(bright, delay_samples)
        right[:delay_samples] = bright[:delay_samples]  # no wrap artifacts
        return left, right

    # ── DSP Primitives ────────────────────────────────────────────────

    def _compress(
        self,
        signal: np.ndarray,
        threshold: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> np.ndarray:
        """Simple feed-forward compressor."""
        attack_coeff = math.exp(-1.0 / (attack_ms * self._sr / 1000))
        release_coeff = math.exp(-1.0 / (release_ms * self._sr / 1000))

        output = np.empty_like(signal)
        env = self._comp_env

        for i in range(len(signal)):
            level = abs(signal[i])
            if level > env:
                env = attack_coeff * env + (1 - attack_coeff) * level
            else:
                env = release_coeff * env + (1 - release_coeff) * level

            if env > threshold:
                gain_reduction = threshold + (env - threshold) / ratio
                gain = gain_reduction / (env + 1e-10)
            else:
                gain = 1.0
            output[i] = signal[i] * gain

        self._comp_env = env
        return output

    def _presence_boost(self, signal: np.ndarray, amount: float) -> np.ndarray:
        """High-frequency presence boost using first-order high-shelf."""
        # Simple high-pass difference technique
        # Extracts high frequencies and adds them back
        alpha = 0.85  # Controls shelf frequency (~3-4kHz at 44.1k)
        hp = np.empty_like(signal)
        hp[0] = signal[0]
        for i in range(1, len(signal)):
            hp[i] = alpha * (hp[i - 1] + signal[i] - signal[i - 1])
        return signal + hp * amount

    @staticmethod
    def _soft_limit(signal: np.ndarray) -> np.ndarray:
        """Soft limiter using tanh — prevents digital clipping."""
        return np.tanh(signal).astype(np.float32)


def main() -> None:
    """CLI entry point — delegates to realtime_hifi_cli.py."""
    from claudio.realtime_hifi_cli import main as _main

    _main()


if __name__ == "__main__":
    main()
