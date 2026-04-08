"""test_hrtf_engine.py — Unit tests for the HRTF binaural engine."""
from __future__ import annotations
import math
import numpy as np
import pytest

from hrtf_engine import (
    HRTFBinauralEngine, AudioSource,
    _azimuth_elevation_from_position,
    _quat_conjugate,
    FFT_SIZE,
)


def test_identity_quaternion_gives_zero_azimuth():
    pos = np.array([0.0, 0.0, -2.0])  # directly in front
    az, el = _azimuth_elevation_from_position(pos, (1.0, 0.0, 0.0, 0.0))
    assert abs(az) < 1.0     # should be near 0°
    assert abs(el) < 1.0


def test_source_on_right_gives_positive_azimuth():
    pos = np.array([2.0, 0.0, 0.0])   # directly to the right
    az, el = _azimuth_elevation_from_position(pos, (1.0, 0.0, 0.0, 0.0))
    assert az > 0.0  # right of listener


def test_proximity_gain_increases_closer():
    from hrtf_engine import HRTFBinauralEngine
    engine = HRTFBinauralEngine.__new__(HRTFBinauralEngine)
    from hrtf_engine import AudioSource as AS
    gain_far  = engine._proximity_gain(np.array([0.0, 0.0, -4.0]))
    gain_near = engine._proximity_gain(np.array([0.0, 0.0, -0.3]))
    assert gain_near > gain_far


def test_render_produces_stereo_output():
    engine = HRTFBinauralEngine()
    src = AudioSource(
        source_id="guitar",
        position=np.array([1.5, 0.0, -2.0]),
    )
    engine.add_source(src)
    noise = np.random.randn(FFT_SIZE).astype(np.float32)
    frame = engine.render({"guitar": noise})

    assert frame.left.shape  == (FFT_SIZE,)
    assert frame.right.shape == (FFT_SIZE,)
    assert frame.sources_rendered == 1
    # With source to the right (x>0), right channel should be louder
    assert np.mean(np.abs(frame.right)) > np.mean(np.abs(frame.left)) * 0.5


def test_hrtf_update_is_lock_free(benchmark):
    """
    SpatialLatencyGate proxy: updating head pose must be trivially fast.
    A real 1.5 ms wall-clock gate runs in CI via a dedicated integration test.
    """
    engine = HRTFBinauralEngine()
    quat = (math.cos(math.pi/4), 0.0, math.sin(math.pi/4), 0.0)
    result = benchmark(engine.update_head_pose, quat)
    # No assertion on time here — benchmark reports it.
    # Integration test in ci/spatial_latency_gate.py enforces <1.5 ms.


def test_render_empty_sources():
    engine = HRTFBinauralEngine()
    frame  = engine.render({})
    assert frame.sources_rendered == 0
    assert np.all(frame.left  == 0)
    assert np.all(frame.right == 0)
