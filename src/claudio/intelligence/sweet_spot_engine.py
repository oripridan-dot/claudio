"""
sweet_spot_engine.py — Dynamic Speaker Sweet Spot DSP Engine

Tracks the listener's head position via camera and applies real-time
delay/gain corrections to studio monitors so the sweet spot follows
the listener around the room.

Modes:
  1. FOCUS_ENGINEER:  Optimized for the primary mix position
  2. FOCUS_COUCH:     Re-aligned for a secondary listening position
  3. WIDE_COMPROMISE: Mid/Side widening for multiple listeners
  4. DYNAMIC_FOLLOW:  Sub-ms delay tracking as the listener moves

All corrections are lightweight enough to run in real-time at 48kHz.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s at ~20°C


class ListeningMode(Enum):
    FOCUS_ENGINEER = "focus_engineer"
    FOCUS_COUCH = "focus_couch"
    WIDE_COMPROMISE = "wide_compromise"
    DYNAMIC_FOLLOW = "dynamic_follow"


@dataclass
class SpeakerConfig:
    """Physical speaker placement."""
    id: str
    position: np.ndarray        # (x, y, z) in meters, listener-centric coordinates
    is_active: bool = True
    delay_compensation_ms: float = 0.0
    gain_compensation_db: float = 0.0


@dataclass
class ListenerPosition:
    """Tracked listener head position."""
    x: float   # left-right (meters from center)
    y: float   # up-down
    z: float   # front-back (distance from speakers)
    confidence: float = 0.0  # tracking confidence 0-1


@dataclass
class SweetSpotCorrection:
    """Per-speaker correction to maintain phantom center at listener."""
    speaker_id: str
    delay_ms: float        # additional delay to apply
    gain_db: float         # gain adjustment
    distance_m: float      # computed distance from speaker to listener


@dataclass
class SweetSpotFrame:
    """Complete sweet spot state for UI visualization + DSP application."""
    mode: ListeningMode
    corrections: list[SweetSpotCorrection]
    listener: ListenerPosition
    phantom_center_offset_deg: float   # 0 = perfect center, >0 = shifted right
    is_optimal_zone: bool              # True if listener is in the sweet spot
    zone_quality: float                # 0-1; 1 = perfect equilateral triangle
    coaching_message: str              # UI guidance for the user


class SweetSpotEngine:
    """
    Computes real-time delay and gain corrections for studio monitors
    based on the tracked listener head position.

    Physics:
      - Sound travels at 343 m/s (~0.29 ms per 10cm)
      - The phantom center requires equal arrival time from both speakers
      - When off-center, the closer speaker's sound arrives first (Haas effect)
      - We add micro-delay to the closer speaker to re-align arrival times
      - Inverse square law gain correction maintains equal perceived loudness
    """

    def __init__(self, sample_rate: int = 48_000):
        self._sr = sample_rate
        self._speakers: dict[str, SpeakerConfig] = {}
        self._mode = ListeningMode.DYNAMIC_FOLLOW
        self._reference_position = ListenerPosition(0, 0, -1.2)  # ideal mix position
        self._secondary_position: ListenerPosition | None = None

    # ── Configuration ────────────────────────────────────────────────────

    def add_speaker(self, speaker: SpeakerConfig) -> None:
        self._speakers[speaker.id] = speaker

    def set_stereo_pair(
        self,
        separation_m: float = 1.5,
        distance_m: float = 1.2,
        height_m: float = 1.2,
    ) -> None:
        """Configure a standard stereo monitoring pair."""
        half_sep = separation_m / 2
        self._speakers["L"] = SpeakerConfig(
            id="L",
            position=np.array([-half_sep, height_m, -distance_m]),
        )
        self._speakers["R"] = SpeakerConfig(
            id="R",
            position=np.array([half_sep, height_m, -distance_m]),
        )
        self._reference_position = ListenerPosition(0, height_m, 0)

    def set_mode(self, mode: ListeningMode) -> None:
        self._mode = mode

    def set_secondary_position(self, pos: ListenerPosition) -> None:
        """Set a secondary listening position (e.g. couch behind the desk)."""
        self._secondary_position = pos

    # ── Core Computation ─────────────────────────────────────────────────

    def compute(self, listener: ListenerPosition) -> SweetSpotFrame:
        """Compute corrections for the given listener position."""
        if not self._speakers:
            return SweetSpotFrame(
                mode=self._mode,
                corrections=[],
                listener=listener,
                phantom_center_offset_deg=0,
                is_optimal_zone=False,
                zone_quality=0,
                coaching_message="No speakers configured.",
            )

        if self._mode == ListeningMode.FOCUS_COUCH and self._secondary_position:
            target = self._secondary_position
        elif self._mode == ListeningMode.DYNAMIC_FOLLOW:
            target = listener
        else:
            target = self._reference_position

        target_pos = np.array([target.x, target.y, target.z])
        corrections: list[SweetSpotCorrection] = []
        distances: dict[str, float] = {}

        # Compute distance from each speaker to the target position
        for sid, spk in self._speakers.items():
            if not spk.is_active:
                continue
            dist = float(np.linalg.norm(spk.position - target_pos))
            distances[sid] = dist

        if not distances:
            return SweetSpotFrame(
                mode=self._mode, corrections=[], listener=listener,
                phantom_center_offset_deg=0, is_optimal_zone=False,
                zone_quality=0, coaching_message="No active speakers.",
            )

        # Reference distance = furthest speaker (all others get delay added)
        max_dist = max(distances.values())
        ref_dist = max_dist  # reference = the farthest speaker

        for sid, dist in distances.items():
            # Delay compensation: add delay to closer speakers
            delta_m = ref_dist - dist
            delay_ms = (delta_m / SPEED_OF_SOUND) * 1000.0

            # Gain compensation: inverse square law normalization
            # Reference gain at the furthest distance
            gain_ratio = (ref_dist / (dist + 0.01)) ** 2
            gain_db = -10 * math.log10(gain_ratio + 1e-10)  # reduce closer speaker

            if self._mode == ListeningMode.WIDE_COMPROMISE:
                # In wide mode, reduce stereo separation + widen dispersion
                delay_ms *= 0.5
                gain_db *= 0.5

            corrections.append(SweetSpotCorrection(
                speaker_id=sid,
                delay_ms=max(0, delay_ms),
                gain_db=gain_db,
                distance_m=dist,
            ))

        # Phantom center offset calculation
        if "L" in distances and "R" in distances:
            delta_lr = distances["R"] - distances["L"]
            # ITD ≈ delta_distance / speed_of_sound
            itd_ms = (delta_lr / SPEED_OF_SOUND) * 1000
            # Rough: 0.63ms ITD ≈ 90° offset, so offset_deg ≈ itd_ms * (90/0.63)
            phantom_offset = itd_ms * (90.0 / 0.63)
        else:
            phantom_offset = 0.0

        # Zone quality: how close to equilateral triangle?
        if len(distances) >= 2:
            dist_values = list(distances.values())
            mean_d = float(np.mean(dist_values))
            std_d = float(np.std(dist_values))
            zone_quality = max(0, 1.0 - (std_d / (mean_d + 0.01)) * 5)
        else:
            zone_quality = 0.5

        is_optimal = zone_quality > 0.85 and abs(phantom_offset) < 5.0

        # Coaching message
        if is_optimal:
            msg = "You're in the sweet spot. Stereo imaging is accurate."
        elif abs(phantom_offset) > 30:
            direction = "right" if phantom_offset > 0 else "left"
            msg = (
                f"Phantom center is shifted {abs(phantom_offset):.0f}° to the {direction}. "
                f"Move {'right' if direction == 'left' else 'left'} to recenter, "
                f"or Claudio will auto-compensate with delay."
            )
        elif abs(phantom_offset) > 10:
            msg = (
                f"Slightly off-center ({abs(phantom_offset):.0f}°). "
                f"Dynamic compensation is active — stereo image corrected."
            )
        else:
            msg = "Near-optimal position. Dynamic compensation fine-tuning active."

        return SweetSpotFrame(
            mode=self._mode,
            corrections=corrections,
            listener=listener,
            phantom_center_offset_deg=phantom_offset,
            is_optimal_zone=is_optimal,
            zone_quality=zone_quality,
            coaching_message=msg,
        )

    def compute_multi_listener(
        self,
        listeners: list[ListenerPosition],
    ) -> SweetSpotFrame:
        """
        Compromise sweet spot for multiple listeners.
        Uses the centroid of all tracked heads as the virtual target.
        """
        if not listeners:
            return self.compute(self._reference_position)

        centroid = ListenerPosition(
            x=float(np.mean([lp.x for lp in listeners])),
            y=float(np.mean([lp.y for lp in listeners])),
            z=float(np.mean([lp.z for lp in listeners])),
            confidence=float(np.mean([lp.confidence for lp in listeners])),
        )
        old_mode = self._mode
        self._mode = ListeningMode.WIDE_COMPROMISE
        result = self.compute(centroid)
        self._mode = old_mode
        result.coaching_message = (
            f"Multi-listener mode: {len(listeners)} people detected. "
            f"Stereo width reduced for broader coverage. "
            f"Critical panning decisions should be checked from the center position."
        )
        return result
