"""
phase_detector.py — Phase Correlation & Polarity Detection Engine

Detects phase cancellation between multi-mic setups (e.g. snare top/bottom,
stereo pair misalignment) and provides corrective recommendations.

Meters:
  - PhaseCorrelationMeter: real-time stereo phase correlation (-1 to +1)
  - PolarityDetector: detects inverted polarity between two channels
  - MicTimingAnalyzer: detects time-of-flight offset between mics

All outputs are structured for UI rendering and hardware control bridging.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PhaseCorrelationFrame:
    """Real-time phase correlation between two channels."""
    correlation: float         # -1.0 (180° out of phase) to +1.0 (perfectly aligned)
    phase_angle_deg: float     # dominant phase angle
    severity: str              # "aligned", "acceptable", "warning", "critical"
    recommendation: str        # human-readable correction advice
    needs_polarity_flip: bool  # True if polarity inversion is the likely fix
    time_offset_samples: int   # estimated sample offset between channels
    time_offset_ms: float      # estimated ms offset


@dataclass
class MultiMicPhaseReport:
    """Phase analysis across all active mic channels."""
    channel_pairs: list[tuple[str, str, PhaseCorrelationFrame]]
    worst_pair: tuple[str, str] | None = None
    overall_coherence: float = 1.0  # 0-1; 1 = all channels perfectly coherent


class PhaseCorrelationMeter:
    """
    Real-time Pearson correlation meter for two audio buffers.

    Operates on rolling windows for continuous display.
    Correlation values:
      +1.0  = perfectly in phase (mono compatible)
       0.0  = no correlation (uncorrelated signals or 90° offset)
      -1.0  = perfectly out of phase (complete cancellation in mono)
    """

    def __init__(self, sample_rate: int = 48_000):
        self._sr = sample_rate

    def analyze(
        self,
        ch1: np.ndarray,
        ch2: np.ndarray,
        ch1_name: str = "CH1",
        ch2_name: str = "CH2",
    ) -> PhaseCorrelationFrame:
        if len(ch1) < 64 or len(ch2) < 64:
            return PhaseCorrelationFrame(
                correlation=0.0, phase_angle_deg=0.0,
                severity="unknown", recommendation="Insufficient audio data.",
                needs_polarity_flip=False, time_offset_samples=0, time_offset_ms=0.0,
            )

        # Ensure same length
        min_len = min(len(ch1), len(ch2))
        ch1 = ch1[:min_len]
        ch2 = ch2[:min_len]

        # Pearson correlation
        ch1_centered = ch1 - np.mean(ch1)
        ch2_centered = ch2 - np.mean(ch2)
        numerator = float(np.sum(ch1_centered * ch2_centered))
        denominator = math.sqrt(
            float(np.sum(ch1_centered ** 2)) * float(np.sum(ch2_centered ** 2))
        ) + 1e-10
        correlation = numerator / denominator

        # Cross-correlation for time offset estimation
        xcorr = np.correlate(ch1_centered, ch2_centered, mode='full')
        center = len(xcorr) // 2
        peak_idx = int(np.argmax(np.abs(xcorr)))
        time_offset_samples = peak_idx - center
        time_offset_ms = (time_offset_samples / self._sr) * 1000

        # Phase angle from correlation + offset
        if abs(correlation) > 0.01:
            phase_angle = math.degrees(math.acos(max(-1, min(1, correlation))))  # 0° to 180°
            if time_offset_samples < 0:
                phase_angle = -phase_angle
        else:
            phase_angle = 90.0  # uncorrelated

        # Severity classification
        needs_flip = False
        if correlation > 0.8:
            severity = "aligned"
            recommendation = f"{ch1_name} and {ch2_name} are well-aligned."
        elif correlation > 0.3:
            severity = "acceptable"
            recommendation = (
                f"Moderate phase offset between {ch1_name} and {ch2_name} "
                f"({time_offset_ms:.1f}ms delay). Consider time-aligning "
                f"by nudging {ch2_name} {abs(time_offset_ms):.1f}ms "
                f"{'earlier' if time_offset_ms > 0 else 'later'}."
            )
        elif correlation > -0.3:
            severity = "warning"
            recommendation = (
                f"Significant phase issues between {ch1_name} and {ch2_name}. "
                f"Phase angle ≈{abs(phase_angle):.0f}°. Check mic distances and "
                f"ensure they're equidistant from the source, or apply "
                f"{abs(time_offset_ms):.1f}ms delay compensation."
            )
        else:
            severity = "critical"
            needs_flip = True
            recommendation = (
                f"SEVERE phase cancellation detected between {ch1_name} and {ch2_name}! "
                f"Correlation: {correlation:.2f} — signals are nearly inverted. "
                f"Flip the polarity (Ø) on {ch2_name} to restore low-end punch."
            )

        return PhaseCorrelationFrame(
            correlation=correlation,
            phase_angle_deg=phase_angle,
            severity=severity,
            recommendation=recommendation,
            needs_polarity_flip=needs_flip,
            time_offset_samples=time_offset_samples,
            time_offset_ms=time_offset_ms,
        )

    def analyze_multi(
        self,
        channels: dict[str, np.ndarray],
    ) -> MultiMicPhaseReport:
        """Analyze phase correlation for all channel pairs."""
        names = list(channels.keys())
        pairs: list[tuple[str, str, PhaseCorrelationFrame]] = []
        worst_correlation = 1.0
        worst_pair: tuple[str, str] | None = None

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                frame = self.analyze(
                    channels[names[i]], channels[names[j]],
                    ch1_name=names[i], ch2_name=names[j],
                )
                pairs.append((names[i], names[j], frame))
                if frame.correlation < worst_correlation:
                    worst_correlation = frame.correlation
                    worst_pair = (names[i], names[j])

        # Overall coherence: mean of all pairwise correlations (mapped to 0-1)
        if pairs:
            mean_corr = float(np.mean([p[2].correlation for p in pairs]))
            coherence = (mean_corr + 1.0) / 2.0  # map [-1,1] to [0,1]
        else:
            coherence = 1.0

        return MultiMicPhaseReport(
            channel_pairs=pairs,
            worst_pair=worst_pair,
            overall_coherence=coherence,
        )


class StereoPhaseScope:
    """
    Generates Lissajous / vectorscope data for stereo phase visualization.

    Output is XY coordinate pairs suitable for Canvas/WebGL rendering:
      - X = (L+R)/2 (Mid)
      - Y = (L-R)/2 (Side)
    A vertical line = mono compatible. Horizontal = out of phase.
    """

    def compute(
        self,
        left: np.ndarray,
        right: np.ndarray,
        downsample: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (x_coords, y_coords) for vectorscope display."""
        min_len = min(len(left), len(right))
        left_ds = left[:min_len:downsample]
        right_ds = right[:min_len:downsample]
        x = (left_ds + right_ds) / 2.0   # Mid
        y = (left_ds - right_ds) / 2.0   # Side
        return x.astype(np.float32), y.astype(np.float32)

    def compute_correlation_history(
        self,
        left: np.ndarray,
        right: np.ndarray,
        window_size: int = 1024,
    ) -> np.ndarray:
        """Rolling correlation over time for a correlation meter trace."""
        min_len = min(len(left), len(right))
        n_windows = min_len // window_size
        history = np.zeros(n_windows)
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            left_w = left[start:end] - np.mean(left[start:end])
            right_w = right[start:end] - np.mean(right[start:end])
            num = float(np.sum(left_w * right_w))
            den = math.sqrt(float(np.sum(left_w ** 2)) * float(np.sum(right_w ** 2))) + 1e-10
            history[i] = num / den
        return history.astype(np.float32)
