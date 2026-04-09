"""
signal_flow_simulator.py — End-to-End Signal Flow Pipeline Simulator

Models the complete Claudio audio path from capture to binaural output
with precise latency and quality measurements at each stage.

Pipeline stages:
  1. Capture  — simulates ADC at capture_sample_rate
  2. Resample — upsamples to render_sample_rate for HRTF
  3. HRTF     — binaural convolution via HRTFBinauralEngine
  4. Output   — simulates DAC buffer at render_sample_rate
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from claudio.hrtf_engine import AudioSource, HRTFBinauralEngine
from claudio.signal_flow_config import SignalFlowConfig
from claudio.signal_flow_metrics import (
    measure_phase_coherence,
    measure_snr,
    measure_thdn,
)


@dataclass
class StageLatency:
    """Latency measurement for a single pipeline stage."""

    name: str
    buffer_latency_ms: float = 0.0
    cpu_time_us: float = 0.0


@dataclass
class QualityMetrics:
    """Measured quality metrics from a simulation run."""

    total_latency_ms: float = 0.0
    snr_db: float = 0.0
    thdn_percent: float = 0.0
    freq_response_deviation_db: float = 0.0
    itd_error_us: float = 0.0
    ild_error_db: float = 0.0
    phase_coherence: float = 0.0
    avg_render_time_us: float = 0.0
    peak_render_time_us: float = 0.0
    stage_latencies: list[StageLatency] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Complete result from a signal flow simulation run."""

    config: SignalFlowConfig
    metrics: QualityMetrics
    passed_all_gates: bool = False
    gate_failures: list[str] = field(default_factory=list)


class SignalFlowSimulator:
    """End-to-end pipeline simulator with quality measurement."""

    def __init__(self, config: SignalFlowConfig | None = None) -> None:
        self._config = config or SignalFlowConfig()
        self._engine = HRTFBinauralEngine(config=self._config)
        self._render_times: list[float] = []

    @property
    def config(self) -> SignalFlowConfig:
        return self._config

    @property
    def engine(self) -> HRTFBinauralEngine:
        return self._engine

    # ── Test Signal Generators ───────────────────────────────────────────

    @staticmethod
    def generate_sine(
        freq_hz: float,
        duration_s: float,
        sample_rate: int,
        amplitude: float = 0.8,
    ) -> np.ndarray:
        t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
        return (np.sin(2 * np.pi * freq_hz * t) * amplitude).astype(np.float32)

    @staticmethod
    def generate_impulse(length: int) -> np.ndarray:
        buf = np.zeros(length, dtype=np.float32)
        buf[0] = 1.0
        return buf

    @staticmethod
    def generate_sweep(
        f_start: float,
        f_end: float,
        duration_s: float,
        sample_rate: int,
        amplitude: float = 0.8,
    ) -> np.ndarray:
        """Logarithmic sine sweep (Farina method)."""
        n = int(sample_rate * duration_s)
        t = np.arange(n, dtype=np.float64) / sample_rate
        phase = (
            2
            * np.pi
            * f_start
            * duration_s
            / np.log(f_end / f_start)
            * (np.exp(t / duration_s * np.log(f_end / f_start)) - 1)
        )
        return (np.sin(phase) * amplitude).astype(np.float32)

    # ── Simulation Runs ──────────────────────────────────────────────────

    def run_sine_test(
        self,
        freq_hz: float = 1000.0,
        duration_s: float = 0.5,
        source_position: np.ndarray | None = None,
        head_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> SimulationResult:
        if source_position is None:
            source_position = np.array([0.0, 0.0, -2.0])
        audio = self.generate_sine(freq_hz, duration_s, self._config.capture_sample_rate)
        return self._run_pipeline(audio, source_position, head_quat, f"sine_{freq_hz}Hz")

    def run_impulse_test(
        self,
        source_position: np.ndarray | None = None,
        head_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> SimulationResult:
        if source_position is None:
            source_position = np.array([0.0, 0.0, -2.0])
        audio = self.generate_impulse(self._config.fft_size * 4)
        return self._run_pipeline(audio, source_position, head_quat, "impulse")

    def run_multi_source_stress(
        self,
        n_sources: int = 8,
        duration_s: float = 0.5,
    ) -> SimulationResult:
        sr = self._config.capture_sample_rate
        block = self._config.fft_size
        n_blocks = int(sr * duration_s) // block

        sources: list[AudioSource] = []
        for i in range(n_sources):
            angle = (i / n_sources) * 2 * math.pi
            pos = np.array([2.0 * math.cos(angle), 0.0, -2.0 * math.sin(angle)])
            src = AudioSource(source_id=f"src_{i}", position=pos)
            self._engine.add_source(src)
            sources.append(src)

        self._render_times = []
        out_l_all, out_r_all = [], []
        for b in range(n_blocks):
            buffers = {}
            for src in sources:
                freq = 220 * (2 ** (sources.index(src) / 12.0))
                t0 = b * block / sr
                t = np.arange(block, dtype=np.float64) / sr + t0
                buffers[src.source_id] = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
            frame = self._engine.render(buffers)
            self._render_times.append(frame.render_time_us)
            out_l_all.append(frame.left)
            out_r_all.append(frame.right)

        for src in sources:
            self._engine.remove_source(src.source_id)

        out_l = np.concatenate(out_l_all)
        out_r = np.concatenate(out_r_all)
        metrics = self._compute_metrics(out_l, out_r, None)
        return self._evaluate_gates(metrics)

    # ── Pipeline Execution ───────────────────────────────────────────────

    def _run_pipeline(
        self,
        audio: np.ndarray,
        source_position: np.ndarray,
        head_quat: tuple[float, float, float, float],
        label: str,
    ) -> SimulationResult:
        block = self._config.fft_size
        src = AudioSource(source_id=label, position=source_position)
        self._engine.add_source(src)
        self._engine.update_head_pose(head_quat)

        if self._config.capture_sample_rate != self._config.render_sample_rate:
            ratio = self._config.render_sample_rate // self._config.capture_sample_rate
            audio = np.repeat(audio, ratio)

        n_blocks = len(audio) // block
        self._render_times = []
        out_l_all, out_r_all = [], []

        for b in range(n_blocks):
            chunk = audio[b * block : (b + 1) * block]
            frame = self._engine.render({label: chunk})
            self._render_times.append(frame.render_time_us)
            out_l_all.append(frame.left)
            out_r_all.append(frame.right)

        self._engine.remove_source(label)

        out_l = np.concatenate(out_l_all) if out_l_all else np.zeros(0, dtype=np.float32)
        out_r = np.concatenate(out_r_all) if out_r_all else np.zeros(0, dtype=np.float32)
        metrics = self._compute_metrics(out_l, out_r, audio[: len(out_l)])
        return self._evaluate_gates(metrics)

    # ── Quality Measurement ──────────────────────────────────────────────

    def _compute_metrics(
        self,
        out_l: np.ndarray,
        out_r: np.ndarray,
        reference: np.ndarray | None,
    ) -> QualityMetrics:
        cfg = self._config
        stages = [
            StageLatency("capture", cfg.capture_latency_ms),
            StageLatency("output", cfg.output_latency_ms),
        ]
        avg_render = float(np.mean(self._render_times)) if self._render_times else 0.0
        peak_render = float(np.max(self._render_times)) if self._render_times else 0.0
        stages.append(StageLatency("hrtf_render", 0.0, avg_render))

        total_lat = cfg.total_buffer_latency_ms + avg_render / 1000.0
        snr = measure_snr(out_l, reference) if reference is not None else 0.0
        thdn = measure_thdn(out_l) if len(out_l) > 0 else 0.0
        coherence = measure_phase_coherence(out_l, out_r)

        return QualityMetrics(
            total_latency_ms=total_lat,
            snr_db=snr,
            thdn_percent=thdn,
            phase_coherence=coherence,
            avg_render_time_us=avg_render,
            peak_render_time_us=peak_render,
            stage_latencies=stages,
        )

    def _evaluate_gates(self, metrics: QualityMetrics) -> SimulationResult:
        q = self._config.quality
        failures: list[str] = []
        if metrics.total_latency_ms > q.max_pipeline_latency_ms:
            failures.append(f"latency {metrics.total_latency_ms:.2f}ms > {q.max_pipeline_latency_ms}ms")
        if metrics.avg_render_time_us > q.max_render_time_per_block_us:
            failures.append(f"render {metrics.avg_render_time_us:.0f}µs > {q.max_render_time_per_block_us}µs")
        return SimulationResult(
            config=self._config,
            metrics=metrics,
            passed_all_gates=len(failures) == 0,
            gate_failures=failures,
        )


def optimize_config(max_iterations: int = 20) -> SimulationResult:
    """Automated parameter sweep to find SOTA-optimal configuration."""
    best_result: SimulationResult | None = None
    best_score = -float("inf")

    sweep_params = [
        {"capture_buffer_size": bs, "output_buffer_size": bs, "fft_size": fft, "hrir_length": hrir}
        for bs in [64, 128, 256]
        for fft in [256, 512]
        for hrir in [128, 256]
    ]

    for _i, params in enumerate(sweep_params[:max_iterations]):
        cfg = SignalFlowConfig(**params)
        sim = SignalFlowSimulator(cfg)
        result = sim.run_sine_test(freq_hz=1000.0, duration_s=0.2)
        score = -result.metrics.total_latency_ms - result.metrics.avg_render_time_us / 1000.0
        if result.passed_all_gates:
            score += 100
        if best_result is None or score > best_score:
            best_score = score
            best_result = result

    return best_result  # type: ignore[return-value]
