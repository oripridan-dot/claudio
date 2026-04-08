"""test_hrtf_engine.py — Unit tests for the HRTF binaural engine."""
from __future__ import annotations

import math

import numpy as np
import pytest

from claudio.hrtf_engine import (
    AudioSource,
    HRTFBinauralEngine,
    _azimuth_elevation_from_position,
)
from claudio.signal_flow_config import SignalFlowConfig


def test_identity_quaternion_gives_zero_azimuth():
    pos = np.array([0.0, 0.0, 2.0])  # directly in front (+Z forward)
    az, el = _azimuth_elevation_from_position(pos, (1.0, 0.0, 0.0, 0.0))
    assert abs(az) < 1.0     # should be near 0°
    assert abs(el) < 1.0


def test_source_on_right_gives_positive_azimuth():
    pos = np.array([2.0, 0.0, 0.0])   # directly to the right
    az, el = _azimuth_elevation_from_position(pos, (1.0, 0.0, 0.0, 0.0))
    assert az > 0.0  # right of listener


def test_proximity_gain_increases_closer():
    engine = HRTFBinauralEngine()
    gain_far  = engine._proximity_gain(np.array([0.0, 0.0, -4.0]))
    gain_near = engine._proximity_gain(np.array([0.0, 0.0, -0.3]))
    assert gain_near > gain_far


def test_render_produces_stereo_output():
    cfg = SignalFlowConfig(fft_size=512, hrir_length=128)
    engine = HRTFBinauralEngine(config=cfg)
    src = AudioSource(
        source_id="guitar",
        position=np.array([1.5, 0.0, -2.0]),
    )
    engine.add_source(src)
    noise = np.random.randn(cfg.fft_size).astype(np.float32)
    frame = engine.render({"guitar": noise})

    assert frame.left.shape  == (cfg.fft_size,)
    assert frame.right.shape == (cfg.fft_size,)
    assert frame.sources_rendered == 1
    # With source to the right (x>0), right channel should have energy
    assert np.mean(np.abs(frame.right)) > 0.0


@pytest.mark.skip(reason="Requires pytest-benchmark fixture")
def test_hrtf_update_is_lock_free(benchmark):
    """
    SpatialLatencyGate proxy: updating head pose must be trivially fast.
    A real 1.5 ms wall-clock gate runs in CI via a dedicated integration test.
    """
    engine = HRTFBinauralEngine()
    quat = (math.cos(math.pi/4), 0.0, math.sin(math.pi/4), 0.0)
    benchmark(engine.update_head_pose, quat)
    # No assertion on time here — benchmark reports it.
    # Integration test in ci/spatial_latency_gate.py enforces <1.5 ms.


def test_render_empty_sources():
    cfg = SignalFlowConfig(fft_size=512)
    engine = HRTFBinauralEngine(config=cfg)
    frame  = engine.render({})
    assert frame.sources_rendered == 0
    assert np.all(frame.left  == 0)
    assert np.all(frame.right == 0)
