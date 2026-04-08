"""
signal_flow_config.py — Claudio Signal Flow Pipeline Configuration

Central configuration dataclass for the entire audio signal flow.
Every tunable parameter lives here — no magic constants scattered
across modules.  The optimisation loop mutates a single Config
instance and re-runs the simulator until SOTA quality gates pass.

Design rationale:
  - Observation / intelligence path runs at 48 kHz (approved Q1)
  - HRTF rendering runs at 192 kHz for maximum spatial resolution
  - Partitioned convolution enabled (approved Q2) for long HRIRs
    without sacrificing latency
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ConvolutionStrategy(Enum):
    """Convolution method used by the HRTF engine."""
    OLA = "overlap_add"                # classic overlap-add — simple, proven
    OLS = "overlap_save"               # overlap-save — fewer artefacts at block edges
    PARTITIONED = "partitioned_ola"    # segmented OLA — long HRIRs at low latency


class HRTFInterpolation(Enum):
    """Method for interpolating between HRTF grid positions."""
    NEAREST = "nearest"      # snap to closest 5° grid point
    BILINEAR = "bilinear"    # spherical bilinear interp between 4 neighbours
    VBAP = "vbap"            # vector-base amplitude panning across 3 neighbours


@dataclass
class QualityTargets:
    """SOTA quality gate thresholds — every metric must pass simultaneously."""
    max_pipeline_latency_ms: float = 15.0
    min_snr_db: float = 120.0                   # 32-bit float theoretical floor
    max_thdn_percent: float = 0.01               # THD+N < 0.01%
    max_freq_response_deviation_db: float = 0.5   # ±0.5 dB flatness 20Hz-20kHz
    max_itd_error_us: float = 5.0                 # ITD accuracy vs Woodworth
    max_ild_error_db: float = 1.0                 # ILD accuracy vs head-shadow
    min_phase_coherence: float = 0.95             # stereo phase correlation floor
    max_render_time_per_block_us: float = 1000.0  # CPU budget per render call
    max_hrtf_crossfade_samples: int = 32          # click-free HRTF update window


@dataclass
class SignalFlowConfig:
    """
    Master pipeline configuration.

    Every parameter has a sensible default tuned for professional
    monitoring on Apple Silicon hardware.  The optimisation sweep
    overrides individual fields and re-runs the simulator.
    """
    # ── Sample Rates ─────────────────────────────────────────────────
    capture_sample_rate: int = 48_000       # ADC / DAW observation tap
    observation_sample_rate: int = 48_000   # intelligence analysis path
    render_sample_rate: int = 192_000       # HRTF convolution internal rate

    # ── Buffer Geometry ──────────────────────────────────────────────
    capture_buffer_size: int = 128          # samples per audio callback
    output_buffer_size: int = 128           # output DAC buffer

    # ── HRTF Engine ──────────────────────────────────────────────────
    fft_size: int = 512                     # OLA / OLS FFT block
    hrir_length: int = 256                  # impulse response samples (was 128)
    hrtf_grid_resolution_deg: float = 5.0   # azimuth/elevation grid spacing
    convolution_strategy: ConvolutionStrategy = ConvolutionStrategy.PARTITIONED
    hrtf_interpolation: HRTFInterpolation = HRTFInterpolation.BILINEAR
    partition_count: int = 2                # segments for partitioned conv
    crossfade_samples: int = 32             # click-free HRTF swap window

    # ── Spatial Model ────────────────────────────────────────────────
    head_radius_m: float = 0.0875           # KEMAR standard
    speed_of_sound_mps: float = 343.0       # at 20°C
    proximity_gain_cap_db: float = 12.0     # max proximity boost
    air_absorption_enabled: bool = True     # ISO 9613-1 frequency-dependent

    # ── Source Limits ────────────────────────────────────────────────
    max_sources: int = 16
    max_distance_m: float = 50.0

    # ── Quality ──────────────────────────────────────────────────────
    quality: QualityTargets = field(default_factory=QualityTargets)

    # ── Derived ──────────────────────────────────────────────────────
    @property
    def capture_latency_ms(self) -> float:
        """One-way capture buffer latency."""
        return (self.capture_buffer_size / self.capture_sample_rate) * 1000.0

    @property
    def output_latency_ms(self) -> float:
        """One-way output buffer latency."""
        return (self.output_buffer_size / self.render_sample_rate) * 1000.0

    @property
    def total_buffer_latency_ms(self) -> float:
        """Combined capture + output buffer latency (no processing time)."""
        return self.capture_latency_ms + self.output_latency_ms

    @property
    def oversampling_factor(self) -> int:
        """How many times render rate exceeds capture rate."""
        return max(1, self.render_sample_rate // self.capture_sample_rate)

    @property
    def partition_segment_length(self) -> int:
        """Samples per partition segment."""
        return max(1, self.hrir_length // self.partition_count)

    @property
    def effective_fft_size(self) -> int:
        """FFT size used by the convolution engine (block + filter - 1)."""
        seg = self.partition_segment_length
        return self._next_power_of_2(self.fft_size + seg - 1)

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p


# ── Pre-built Configurations ─────────────────────────────────────────────

def low_latency_config() -> SignalFlowConfig:
    """Optimised for minimum latency — 64-sample buffers, shorter HRIR."""
    return SignalFlowConfig(
        capture_buffer_size=64,
        output_buffer_size=64,
        hrir_length=128,
        fft_size=256,
        partition_count=2,
        convolution_strategy=ConvolutionStrategy.PARTITIONED,
    )


def high_fidelity_config() -> SignalFlowConfig:
    """Optimised for maximum spatial accuracy — longer HRIR, finer grid."""
    return SignalFlowConfig(
        capture_buffer_size=256,
        output_buffer_size=256,
        hrir_length=512,
        fft_size=1024,
        partition_count=4,
        hrtf_grid_resolution_deg=2.5,
        hrtf_interpolation=HRTFInterpolation.VBAP,
    )


def balanced_config() -> SignalFlowConfig:
    """Default balanced configuration — the recommended starting point."""
    return SignalFlowConfig()
