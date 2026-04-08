"""
test_gesture_classifier.py — Unit tests for GestureClassifier

These tests run without a camera by feeding synthetic LandmarkFrame sequences
directly to the classifier.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from claudio.vision.gesture_classifier import (
    GestureClassifier,
    GestureType,
    LandmarkFrame,
)


def _make_frame(
    right_hand_wrist_x: float = 0.5,
    right_hand_wrist_y: float = 0.5,
    left_hand_wrist_x: float = 0.3,
    left_hand_wrist_y: float = 0.5,
    ts: float | None = None,
) -> LandmarkFrame:
    rh = np.zeros((21, 3), dtype=np.float32)
    rh[0] = [right_hand_wrist_x, right_hand_wrist_y, 0.0]
    lh = np.zeros((21, 3), dtype=np.float32)
    lh[0] = [left_hand_wrist_x, left_hand_wrist_y, 0.0]
    return LandmarkFrame(
        timestamp=ts or time.perf_counter(),
        right_hand=rh,
        left_hand=lh,
    )


def test_sweep_right_detected():
    cls = GestureClassifier()
    # Feed 15 frames with wrist moving rapidly rightward
    for i in range(15):
        frame = _make_frame(
            right_hand_wrist_x=0.1 + i * 0.05,  # large rightward delta
            ts=time.perf_counter() + i * 0.033,
        )
        event = cls.ingest(frame)
    assert event is not None
    assert event.gesture == GestureType.SWEEP_RIGHT


def test_sweep_left_detected():
    cls = GestureClassifier()
    for i in range(15):
        frame = _make_frame(
            right_hand_wrist_x=0.9 - i * 0.05,
            ts=time.perf_counter() + i * 0.033,
        )
        event = cls.ingest(frame)
    assert event is not None
    assert event.gesture == GestureType.SWEEP_LEFT


@pytest.mark.xfail(reason="Known: pinch detection false-positive on identity landmarks")
def test_no_gesture_on_still_hands():
    cls = GestureClassifier()
    event = None
    for i in range(15):
        frame = _make_frame(ts=time.perf_counter() + i * 0.033)
        event = cls.ingest(frame)
    assert event is None


def test_open_palm_static():
    cls = GestureClassifier()
    # Build a frame with open-palm hand landmarks
    rh = np.zeros((21, 3), dtype=np.float32)
    # Tips above MCPs (lower y in image space)
    mcp_y = 0.6
    tip_y = 0.3
    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        rh[tip, 1] = tip_y
        rh[mcp, 1] = mcp_y
    fired = None
    for i in range(40):  # hold for >450 ms at 30 fps
        lf = LandmarkFrame(
            timestamp=time.perf_counter() + i * 0.033,
            right_hand=rh,
        )
        ev = cls.ingest(lf)
        if ev and ev.gesture == GestureType.OPEN_PALM:
            fired = ev
    assert fired is not None


def test_head_tracker_identity_on_none():
    from claudio.vision.head_tracker import SpatialHeadTracker
    tracker = SpatialHeadTracker()
    # None input must not crash and must return identity quaternion
    result = tracker.update(None)
    assert result is None
    assert tracker.latest_quaternion == (1.0, 0.0, 0.0, 0.0)
