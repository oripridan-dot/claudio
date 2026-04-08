"""
gesture_classifier.py — Real-Time Gesture Classification Engine

Uses a sliding window over MediaPipe landmark sequences to classify:
  1. StaticGestures  — pose held for >500 ms
  2. DynamicGestures — directional movement above a velocity threshold
  3. ProximityGestures — relative distance / lean toward camera

The classifier is intentionally lightweight (CPU-only, <4ms per frame)
so it never adds latency to the audio engine.

Architecture:
  - StaticGesture: rule-based finger angle + joint distance heuristics
  - DynamicGesture: sliding-window velocity vector over 15 frames
  - ProximityGesture: normalised z-depth delta + head-pose yaw/pitch from
                      face mesh landmarks
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class GestureType(Enum):
    NONE                = auto()
    SWEEP_RIGHT         = auto()
    SWEEP_LEFT          = auto()
    RAISE_BOTH_HANDS    = auto()
    LOWER_BOTH_HANDS    = auto()
    OPEN_PALM           = auto()
    FIST                = auto()
    HEAD_LEAN_LEFT      = auto()
    HEAD_LEAN_RIGHT     = auto()
    PINCH               = auto()
    TWO_HAND_EXPAND     = auto()
    HEAD_RAISE          = auto()   # reverb / space increase
    HEAD_LOWER          = auto()   # dry / close increase


@dataclass
class GestureEvent:
    gesture:    GestureType
    confidence: float           # 0.0–1.0
    timestamp:  float
    hand:       str = "both"    # "left", "right", "both", "head"
    magnitude:  float = 0.0     # normalised gesture magnitude


@dataclass
class LandmarkFrame:
    """
    Single camera frame worth of MediaPipe landmark data.
    All coordinates are normalised [0.0–1.0] relative to image dimensions.
    z is depth (negative = closer to camera).
    """
    timestamp: float
    # Pose: 33 landmarks × 3 (x, y, z)
    pose: Optional[np.ndarray] = None            # shape (33, 3)
    # Hands: 21 landmarks × 3
    left_hand:  Optional[np.ndarray] = None      # shape (21, 3)
    right_hand: Optional[np.ndarray] = None      # shape (21, 3)
    # Face: 468 landmarks × 3 (used for head pose)
    face: Optional[np.ndarray] = None            # shape (468, 3)


class GestureClassifier:
    """
    Stateful sliding-window gesture classifier.
    Feed it LandmarkFrames; retrieve GestureEvents.

    Thread-safe: designed to run on the camera thread.
    Results are pushed to a thread-safe deque consumed by the routing engine.
    """

    WINDOW          = 15        # frames (~500 ms at 30 fps)
    VEL_THRESHOLD   = 0.04      # normalised units/frame for dynamic gestures
    STATIC_HOLD_MS  = 450       # ms a static gesture must be held
    PINCH_DIST      = 0.06      # normalised distance for pinch detection
    HEAD_YAW_DEG    = 14.0      # degrees of yaw for head-lean detection

    def __init__(self) -> None:
        self._frame_buffer: deque[LandmarkFrame] = deque(maxlen=self.WINDOW)
        self._static_start: dict[GestureType, float] = {}
        self.events: deque[GestureEvent] = deque(maxlen=64)

    # ── Public API ────────────────────────────────────────────────────────

    def ingest(self, frame: LandmarkFrame) -> Optional[GestureEvent]:
        """Feed one landmark frame. Returns a GestureEvent if one fires."""
        self._frame_buffer.append(frame)
        event = self._classify()
        if event:
            self.events.append(event)
        return event

    # ── Classification ────────────────────────────────────────────────────

    def _classify(self) -> Optional[GestureEvent]:
        if len(self._frame_buffer) < 2:
            return None

        frame = self._frame_buffer[-1]
        checks = [
            self._check_dynamic_sweep(frame),
            self._check_dynamic_vertical(frame),
            self._check_head_lean(frame),
            self._check_head_vertical(frame),
            self._check_pinch(frame),
            self._check_expand(frame),
            self._check_static_palm(frame),
            self._check_static_fist(frame),
        ]
        for event in checks:
            if event:
                return event
        return None

    def _hand_velocity(self, hand_attr: str, axis: int) -> float:
        """Compute mean velocity of wrist (landmark 0) along `axis` over window."""
        positions = []
        for f in self._frame_buffer:
            lm = getattr(f, hand_attr)
            if lm is not None:
                positions.append(lm[0, axis])
        if len(positions) < 2:
            return 0.0
        deltas = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        return float(np.mean(deltas))

    def _check_dynamic_sweep(self, frame: LandmarkFrame) -> Optional[GestureEvent]:
        """Detect lateral (x-axis) wrist sweep on either hand."""
        for hand_attr, hand_label in [("right_hand", "right"), ("left_hand", "left")]:
            vx = self._hand_velocity(hand_attr, axis=0)  # x axis
            if abs(vx) > self.VEL_THRESHOLD:
                gt = GestureType.SWEEP_RIGHT if vx > 0 else GestureType.SWEEP_LEFT
                return GestureEvent(
                    gesture=gt,
                    confidence=min(1.0, abs(vx) / (self.VEL_THRESHOLD * 3)),
                    timestamp=frame.timestamp,
                    hand=hand_label,
                    magnitude=abs(vx),
                )
        return None

    def _check_dynamic_vertical(self, frame: LandmarkFrame) -> Optional[GestureEvent]:
        """Detect both-hands vertical movement."""
        vy_r = self._hand_velocity("right_hand", axis=1)
        vy_l = self._hand_velocity("left_hand",  axis=1)
        # Both hands must move in the same vertical direction
        if abs(vy_r) > self.VEL_THRESHOLD and abs(vy_l) > self.VEL_THRESHOLD:
            same_dir = (vy_r < 0 and vy_l < 0) or (vy_r > 0 and vy_l > 0)
            if same_dir:
                # MediaPipe y=0 is top, so negative = raise
                gt = (GestureType.RAISE_BOTH_HANDS if vy_r < 0
                      else GestureType.LOWER_BOTH_HANDS)
                mag = (abs(vy_r) + abs(vy_l)) / 2
                return GestureEvent(
                    gesture=gt, confidence=min(1.0, mag / (self.VEL_THRESHOLD * 3)),
                    timestamp=frame.timestamp, hand="both", magnitude=mag,
                )
        return None

    def _check_head_lean(self, frame: LandmarkFrame) -> Optional[GestureEvent]:
        """Detect head yaw (lean left/right) from face mesh landmarks."""
        if frame.face is None or len(frame.face) < 468:
            # Fall back to pose nose/ear if no face mesh
            if frame.pose is None:
                return None
            nose_x   = frame.pose[0, 0]
            left_ear = frame.pose[7, 0]
            right_ear = frame.pose[8, 0]
            ear_span = abs(right_ear - left_ear) + 1e-6
            yaw_proxy = (nose_x - (left_ear + right_ear) / 2) / ear_span
        else:
            # Face mesh: tip of nose=1, left temple=234, right temple=454
            nose_x    = frame.face[1, 0]
            left_t    = frame.face[234, 0]
            right_t   = frame.face[454, 0]
            span      = abs(right_t - left_t) + 1e-6
            yaw_proxy = (nose_x - (left_t + right_t) / 2) / span

        # yaw_proxy > 0 = head turned right in image (lean left from user perspective)
        threshold = self.HEAD_YAW_DEG / 90.0
        if yaw_proxy > threshold:
            return GestureEvent(
                gesture=GestureType.HEAD_LEAN_LEFT,
                confidence=min(1.0, yaw_proxy / (threshold * 2)),
                timestamp=frame.timestamp, hand="head",
                magnitude=abs(yaw_proxy),
            )
        if yaw_proxy < -threshold:
            return GestureEvent(
                gesture=GestureType.HEAD_LEAN_RIGHT,
                confidence=min(1.0, abs(yaw_proxy) / (threshold * 2)),
                timestamp=frame.timestamp, hand="head",
                magnitude=abs(yaw_proxy),
            )
        return None

    def _check_head_vertical(self, frame: LandmarkFrame) -> Optional[GestureEvent]:
        """Detect head pitch (nod up/down) from pose nose landmark."""
        if frame.pose is None:
            return None
        vy = self._hand_velocity("pose", 1) if hasattr(self, "_pose_vel_hack") else 0.0
        # Use nose y across recent frames
        nose_ys = []
        for f in self._frame_buffer:
            if f.pose is not None:
                nose_ys.append(f.pose[0, 1])
        if len(nose_ys) < 2:
            return None
        vy = np.mean(np.diff(nose_ys))
        threshold = self.VEL_THRESHOLD * 0.5
        if vy < -threshold:
            return GestureEvent(
                gesture=GestureType.HEAD_RAISE,
                confidence=min(1.0, abs(vy) / (threshold * 3)),
                timestamp=frame.timestamp, hand="head", magnitude=abs(vy),
            )
        if vy > threshold:
            return GestureEvent(
                gesture=GestureType.HEAD_LOWER,
                confidence=min(1.0, abs(vy) / (threshold * 3)),
                timestamp=frame.timestamp, hand="head", magnitude=abs(vy),
            )
        return None

    def _check_pinch(self, frame: LandmarkFrame) -> Optional[GestureEvent]:
        """Index-thumb pinch on either hand."""
        for hand_attr, hand_label in [("right_hand", "right"), ("left_hand", "left")]:
            lm = getattr(frame, hand_attr)
            if lm is None:
                continue
            # Landmark 4 = thumb tip, landmark 8 = index tip
            dist = float(np.linalg.norm(lm[4] - lm[8]))
            if dist < self.PINCH_DIST:
                return GestureEvent(
                    gesture=GestureType.PINCH,
                    confidence=1.0 - dist / self.PINCH_DIST,
                    timestamp=frame.timestamp, hand=hand_label,
                    magnitude=1.0 - dist / self.PINCH_DIST,
                )
        return None

    def _check_expand(self, frame: LandmarkFrame) -> Optional[GestureEvent]:
        """Both hands moving apart horizontally (stereo widen)."""
        ll = frame.left_hand
        rl = frame.right_hand
        if ll is None or rl is None:
            return None
        lx = ll[0, 0]
        rx = rl[0, 0]
        # Look for expanding distance across frames
        if len(self._frame_buffer) < 5:
            return None
        prev = self._frame_buffer[-5]
        pl = getattr(prev, "left_hand")
        pr = getattr(prev, "right_hand")
        if pl is None or pr is None:
            return None
        prev_span = abs(pr[0, 0] - pl[0, 0])
        curr_span = abs(rx - lx)
        delta = curr_span - prev_span
        if delta > self.VEL_THRESHOLD * 2:
            return GestureEvent(
                gesture=GestureType.TWO_HAND_EXPAND,
                confidence=min(1.0, delta / (self.VEL_THRESHOLD * 5)),
                timestamp=frame.timestamp, hand="both", magnitude=delta,
            )
        return None

    def _check_static_palm(self, frame: LandmarkFrame) -> Optional[GestureEvent]:
        """Open palm held for STATIC_HOLD_MS ms → bypass insert."""
        for hand_attr, hand_label in [("right_hand", "right"), ("left_hand", "left")]:
            lm = getattr(frame, hand_attr)
            if lm is None:
                continue
            if self._is_open_palm(lm):
                key = GestureType.OPEN_PALM
                if key not in self._static_start:
                    self._static_start[key] = frame.timestamp
                elif (frame.timestamp - self._static_start[key]) * 1000 > self.STATIC_HOLD_MS:
                    del self._static_start[key]
                    return GestureEvent(
                        gesture=key, confidence=0.95,
                        timestamp=frame.timestamp, hand=hand_label, magnitude=1.0,
                    )
            else:
                self._static_start.pop(GestureType.OPEN_PALM, None)
        return None

    def _check_static_fist(self, frame: LandmarkFrame) -> Optional[GestureEvent]:
        """Closed fist held for STATIC_HOLD_MS ms → mute channel."""
        for hand_attr, hand_label in [("right_hand", "right"), ("left_hand", "left")]:
            lm = getattr(frame, hand_attr)
            if lm is None:
                continue
            if self._is_fist(lm):
                key = GestureType.FIST
                if key not in self._static_start:
                    self._static_start[key] = frame.timestamp
                elif (frame.timestamp - self._static_start[key]) * 1000 > self.STATIC_HOLD_MS:
                    del self._static_start[key]
                    return GestureEvent(
                        gesture=key, confidence=0.95,
                        timestamp=frame.timestamp, hand=hand_label, magnitude=1.0,
                    )
            else:
                self._static_start.pop(GestureType.FIST, None)
        return None

    # ── Static gesture heuristics ─────────────────────────────────────────

    @staticmethod
    def _is_open_palm(lm: np.ndarray) -> bool:
        """
        Open palm: all four finger tips (8,12,16,20) above their MCPs (5,9,13,17).
        MediaPipe y=0 is top, so "above" means lower y value.
        """
        for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            if lm[tip, 1] >= lm[mcp, 1]:
                return False
        return True

    @staticmethod
    def _is_fist(lm: np.ndarray) -> bool:
        """
        Fist: all four finger tips (8,12,16,20) below their PIPs (6,10,14,18).
        (Curled fingers: tip y > pip y in image space.)
        """
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if lm[tip, 1] <= lm[pip, 1]:
                return False
        return True
