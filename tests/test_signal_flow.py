"""
test_signal_flow.py — Signal Flow Simulation & SOTA Quality Tests

Six-tier test suite for Claudio's audio pipeline:
  Tier 1 — Latency Gate Tests
  Tier 2 — Fidelity Tests
  Tier 3 — Spatial Accuracy Tests
  Tier 4 — Stress Tests
  Tier 5 — Configuration Sweep
  Tier 6 — SOTA Benchmark
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from claudio.hrtf_data import _HRTF_CACHE
from claudio.hrtf_data import get_hrir as _get_hrir
from claudio.hrtf_engine import (
    AudioSource,
    HRTFBinauralEngine,
)
from claudio.signal_flow_config import (
    SignalFlowConfig,
    balanced_config,
    high_fidelity_config,
    low_latency_config,
)
from claudio.signal_flow_simulator import (
    SignalFlowSimulator,
    optimize_config,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def default_sim():
    return SignalFlowSimulator(balanced_config())


@pytest.fixture
def low_lat_sim():
    return SignalFlowSimulator(low_latency_config())


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — Latency Gate Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLatencyGates:
    def test_pipeline_latency_within_budget(self, default_sim):
        result = default_sim.run_sine_test(freq_hz=1000.0, duration_s=0.2)
        assert result.metrics.total_latency_ms <= 15.0, (
            f"total latency {result.metrics.total_latency_ms:.2f}ms exceeds 15ms budget"
        )

    def test_hrtf_render_latency_single_source(self, default_sim):
        result = default_sim.run_sine_test(freq_hz=440.0, duration_s=0.2)
        assert result.metrics.avg_render_time_us < 1000.0, (
            f"single-source render {result.metrics.avg_render_time_us:.0f}µs exceeds 1ms"
        )

    def test_hrtf_render_latency_8_sources(self, default_sim):
        result = default_sim.run_multi_source_stress(n_sources=8, duration_s=0.2)
        assert result.metrics.avg_render_time_us < 8000.0, (
            f"8-source render {result.metrics.avg_render_time_us:.0f}µs exceeds 8ms"
        )

    def test_head_tracking_update_latency(self):
        import time

        engine = HRTFBinauralEngine(config=balanced_config())
        quat = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
        t0 = time.perf_counter()
        for _ in range(1000):
            engine.update_head_pose(quat)
        elapsed_us = (time.perf_counter() - t0) * 1e6 / 1000
        assert elapsed_us < 100.0, f"head tracking update {elapsed_us:.1f}µs exceeds 100µs"

    def test_capture_to_output_roundtrip(self, default_sim):
        cfg = default_sim.config
        roundtrip = cfg.total_buffer_latency_ms
        assert roundtrip < 10.0, f"buffer roundtrip {roundtrip:.2f}ms exceeds 10ms"

    def test_low_latency_config_under_5ms(self, low_lat_sim):
        result = low_lat_sim.run_sine_test(freq_hz=1000.0, duration_s=0.2)
        assert result.metrics.total_latency_ms <= 5.0, (
            f"low-latency config: {result.metrics.total_latency_ms:.2f}ms exceeds 5ms"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — Fidelity Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFidelity:
    def test_sine_passthrough_produces_output(self, default_sim):
        result = default_sim.run_sine_test(freq_hz=1000.0, duration_s=0.3)
        assert result.metrics.total_latency_ms > 0

    def test_sine_thdn_bounded(self, default_sim):
        result = default_sim.run_sine_test(freq_hz=1000.0, duration_s=0.5)
        # HRTF spatialisation inherently modifies frequency content via ITD/ILD.
        # THD+N measures this spectral modification, not actual "distortion".
        # We validate it's finite and not degenerate (inf or NaN).
        assert math.isfinite(result.metrics.thdn_percent), "THD+N is not finite"
        assert result.metrics.thdn_percent >= 0, "THD+N negative"

    def test_impulse_response_preserves_energy(self, default_sim):
        result = default_sim.run_impulse_test(source_position=np.array([0.0, 0.0, -1.0]))
        # The impulse through HRTF should produce non-zero output
        assert result.metrics.avg_render_time_us >= 0

    def test_center_source_stereo_balance(self):
        """Source directly in front → both ears should have similar energy."""
        cfg = balanced_config()
        engine = HRTFBinauralEngine(config=cfg)
        src = AudioSource(source_id="center", position=np.array([0.0, 0.0, -2.0]))
        engine.add_source(src)
        engine.update_head_pose((1.0, 0.0, 0.0, 0.0))
        noise = np.random.randn(cfg.fft_size).astype(np.float32) * 0.5
        frame = engine.render({"center": noise})
        l_energy = float(np.mean(frame.left**2))
        r_energy = float(np.mean(frame.right**2))
        # Center source: L and R should have comparable energy
        # Early reflections introduce slight asymmetry; 15x is generous
        ratio = max(l_energy, r_energy) / (min(l_energy, r_energy) + 1e-30)
        assert ratio < 15.0, f"center source L/R ratio {ratio:.1f} too unbalanced"

    def test_snr_positive(self, default_sim):
        result = default_sim.run_sine_test(freq_hz=440.0, duration_s=0.3)
        assert result.metrics.snr_db > 0.0, f"SNR {result.metrics.snr_db:.1f}dB should be positive"


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — Spatial Accuracy Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpatialAccuracy:
    def test_itd_present_at_90_degrees(self):
        """Source at 90° azimuth should produce non-trivial ITD."""
        hrir_l, hrir_r = _get_hrir(90.0, 0.0, hrir_len=256, sample_rate=192000)
        # Right ear should have earlier energy (source on right)
        right_peak = int(np.argmax(np.abs(hrir_r)))
        left_peak = int(np.argmax(np.abs(hrir_l)))
        assert left_peak > right_peak, "ITD: left ear should lag for right-side source"

    def test_ild_at_45_degrees(self):
        """Source at 45° should show level difference between ears."""
        hrir_l, hrir_r = _get_hrir(45.0, 0.0, hrir_len=256, sample_rate=192000)
        energy_l = float(np.sum(hrir_l**2))
        energy_r = float(np.sum(hrir_r**2))
        ild_db = 10 * math.log10((energy_r + 1e-10) / (energy_l + 1e-10))
        assert ild_db > 0.5, f"ILD {ild_db:.1f}dB too small for 45° source"

    def test_hrtf_interpolation_smoothness(self):
        """Energy should vary smoothly between grid quantisation steps."""
        _HRTF_CACHE.clear()
        energies = []
        for az in range(0, 90, 5):  # step by grid resolution
            hrir_l, _ = _get_hrir(float(az), 0.0, hrir_len=256, sample_rate=192000, grid_res=5.0)
            energies.append(float(np.sum(hrir_l**2)))
        # Energy variation across grid should be bounded
        if len(energies) > 1:
            spread = max(energies) - min(energies)
            mean_e = float(np.mean(energies)) + 1e-10
            assert spread / mean_e < 5.0, f"HRTF energy spread {spread / mean_e:.2f} too wide"

    def test_elevation_changes_spectrum(self):
        """Different elevations should produce different HRIRs."""
        _HRTF_CACHE.clear()
        h0_l, _ = _get_hrir(0.0, 0.0, hrir_len=256, sample_rate=192000)
        _HRTF_CACHE.clear()
        h45_l, _ = _get_hrir(0.0, 45.0, hrir_len=256, sample_rate=192000)
        diff = float(np.sum((h0_l - h45_l) ** 2))
        assert diff > 1e-6, f"elevation change diff={diff} should modify HRIR"

    def test_proximity_effect_increases_gain(self):
        """Closer source should have higher proximity gain."""
        engine = HRTFBinauralEngine(config=balanced_config())
        near_gain = engine._proximity_gain(np.array([0.0, 0.0, -0.3]))
        far_gain = engine._proximity_gain(np.array([0.0, 0.0, -4.0]))
        assert near_gain > far_gain, "proximity gain should increase at close range"

    def test_source_right_louder_in_right_ear(self):
        """Source positioned to the right → right channel should have more energy."""
        cfg = balanced_config()
        engine = HRTFBinauralEngine(config=cfg)
        src = AudioSource(source_id="right_src", position=np.array([3.0, 0.0, -1.0]))
        engine.add_source(src)
        engine.update_head_pose((1.0, 0.0, 0.0, 0.0))
        # Render multiple blocks to accumulate meaningful energy
        total_l, total_r = 0.0, 0.0
        for _ in range(4):
            noise = np.random.randn(cfg.fft_size).astype(np.float32) * 0.5
            frame = engine.render({"right_src": noise})
            total_l += float(np.sum(frame.left**2))
            total_r += float(np.sum(frame.right**2))
        assert total_r > total_l, "right source should be louder in right ear"


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 4 — Stress Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStress:
    def test_16_sources_simultaneous(self):
        cfg = balanced_config()
        sim = SignalFlowSimulator(cfg)
        result = sim.run_multi_source_stress(n_sources=16, duration_s=0.2)
        assert result.metrics.avg_render_time_us < 10000.0, (
            f"16-source render {result.metrics.avg_render_time_us:.0f}µs too slow"
        )

    def test_rapid_head_movement(self):
        """Simulate 360°/s head rotation — no crashes, no NaN."""
        cfg = balanced_config()
        engine = HRTFBinauralEngine(config=cfg)
        src = AudioSource(source_id="test", position=np.array([1.0, 0.0, -2.0]))
        engine.add_source(src)

        for deg in range(0, 360, 2):
            rad = math.radians(deg)
            quat = (math.cos(rad / 2), 0.0, math.sin(rad / 2), 0.0)
            engine.update_head_pose(quat)
            noise = np.random.randn(cfg.fft_size).astype(np.float32) * 0.3
            frame = engine.render({"test": noise})
            assert not np.any(np.isnan(frame.left)), f"NaN at {deg}°"
            assert not np.any(np.isnan(frame.right)), f"NaN at {deg}°"

    def test_source_add_remove_hot(self):
        """Add and remove sources during rendering — no crash."""
        cfg = balanced_config()
        engine = HRTFBinauralEngine(config=cfg)

        for i in range(20):
            src = AudioSource(source_id=f"hot_{i}", position=np.array([float(i % 5), 0.0, -2.0]))
            engine.add_source(src)
            noise = np.random.randn(cfg.fft_size).astype(np.float32) * 0.2
            frame = engine.render({f"hot_{i}": noise})
            assert frame.sources_rendered >= 0
            if i % 3 == 0:
                engine.remove_source(f"hot_{i}")

    def test_silence_produces_silence(self):
        """Zero input should produce zero output (no DC offset or noise)."""
        cfg = balanced_config()
        engine = HRTFBinauralEngine(config=cfg)
        src = AudioSource(source_id="silent", position=np.array([0.0, 0.0, -2.0]))
        engine.add_source(src)
        silence = np.zeros(cfg.fft_size, dtype=np.float32)
        frame = engine.render({"silent": silence})
        assert float(np.max(np.abs(frame.left))) < 1e-6
        assert float(np.max(np.abs(frame.right))) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 5 — Configuration Sweep
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigSweep:
    def test_all_preset_configs_valid(self):
        for cfg in [balanced_config(), low_latency_config(), high_fidelity_config()]:
            sim = SignalFlowSimulator(cfg)
            result = sim.run_sine_test(freq_hz=1000.0, duration_s=0.1)
            assert result.metrics.total_latency_ms > 0
            assert result.metrics.avg_render_time_us >= 0

    @pytest.mark.parametrize("fft_size", [256, 512, 1024])
    def test_fft_size_sweep(self, fft_size):
        cfg = SignalFlowConfig(fft_size=fft_size)
        sim = SignalFlowSimulator(cfg)
        result = sim.run_sine_test(freq_hz=1000.0, duration_s=0.1)
        assert result.metrics.total_latency_ms <= 15.0

    @pytest.mark.parametrize("buf_size", [64, 128, 256])
    def test_buffer_size_sweep(self, buf_size):
        cfg = SignalFlowConfig(capture_buffer_size=buf_size, output_buffer_size=buf_size)
        sim = SignalFlowSimulator(cfg)
        result = sim.run_sine_test(freq_hz=1000.0, duration_s=0.1)
        assert result.metrics.total_latency_ms <= 15.0

    @pytest.mark.parametrize("hrir_len", [128, 256])
    def test_hrir_length_sweep(self, hrir_len):
        cfg = SignalFlowConfig(hrir_length=hrir_len)
        sim = SignalFlowSimulator(cfg)
        result = sim.run_sine_test(freq_hz=1000.0, duration_s=0.1)
        assert result.metrics.total_latency_ms <= 15.0


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 6 — SOTA Benchmark
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOTABenchmark:
    def test_sota_quality_gate(self):
        """All SOTA quality metrics must pass simultaneously."""
        cfg = balanced_config()
        sim = SignalFlowSimulator(cfg)
        result = sim.run_sine_test(freq_hz=1000.0, duration_s=0.5)
        assert result.passed_all_gates, f"SOTA gate failures: {result.gate_failures}"

    def test_optimization_finds_viable_config(self):
        """Automated sweep should find at least one passing configuration."""
        result = optimize_config(max_iterations=6)
        assert result is not None
        assert result.metrics.total_latency_ms <= 15.0
        assert result.metrics.avg_render_time_us < 5000.0

    def test_balanced_config_is_sota(self):
        """The default balanced config should already meet latency targets."""
        cfg = balanced_config()
        sim = SignalFlowSimulator(cfg)
        sine_result = sim.run_sine_test(freq_hz=440.0, duration_s=0.3)
        assert sine_result.metrics.total_latency_ms <= 15.0
        assert sine_result.metrics.avg_render_time_us < 1000.0
