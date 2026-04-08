"""
semantic_metering.py — Semantic Metering Engine

Replaces traditional dBFS/LUFS meters with performance-aware meters that
visualise musical intent, not electrical voltage.

Meters:
  1. PocketRadar       — drummer/bassist groove pocket (ITD between players)
  2. TopographicFreqMap — 3D frequency collision map (mud detection)
  3. PerformanceCoach  — real-time micro-timing and pick-attack guidance

The AcousticEnvironmentAdvisor lives in acoustic_advisor.py (extracted for
single-responsibility compliance).

All meters are stateless processors — they receive audio/MIDI buffers and
return visualisation-ready data structures.  No side effects.
The UI (claudio-studio-app) consumes these data structures to render the
real-time holographic displays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Re-export for backward compatibility
from .acoustic_advisor import AcousticAdvice, AcousticEnvironmentAdvisor

# ─── 1. Pocket Radar ─────────────────────────────────────────────────────────

@dataclass
class PocketRadarFrame:
    """
    Visual data for the Pocket Radar meter.
    Describes the bassist's rhythmic relationship to the drummer's kick grid.

    UI rendering:
      - Circular radar (0° = perfect on-beat, ±°= deviation)
      - "Aura" around the radar point indicates consistency
      - Colour: green=locked, amber=expressive, red=erratic
    """
    bassist_deviation_ms:   float   # mean offset from drummer's kick grid
    bassist_consistency_ms: float   # std-dev of offset (expressiveness vs. error)
    drummer_grid_bpm:       float
    pocket_score:           float   # 0.0 (erratic) – 1.0 (locked)
    # Normalised angle for radar display (-1.0=rushing, +1.0=dragging)
    radar_angle:            float
    radar_magnitude:        float   # 0.0 (centre) – 1.0 (edge)
    aura_colour:            str     # "green", "amber", "red"


class PocketRadar:
    """
    Computes the rhythmic pocket between a bass track and a drum reference.

    Inputs:
      - bass_onsets_sec: list of detected bass note onset times (seconds)
      - kick_onsets_sec: list of detected kick drum onsets (seconds)
      - bpm: current session BPM

    The pocket score follows a Gaussian model: a deviation of ±6 ms is
    considered "locked in" (score > 0.9), ±20 ms is "expressive groove",
    and beyond ±35 ms starts sounding unintentional.
    """

    LOCKED_MS    = 6.0    # within this = perfectly pocket
    EXPRESSIVE_MS = 20.0  # within this = intentional groove feel
    ERRATIC_MS   = 35.0   # beyond this = likely unintentional

    def compute(
        self,
        bass_onsets_sec: list[float],
        kick_onsets_sec: list[float],
        bpm: float,
    ) -> PocketRadarFrame | None:
        if not bass_onsets_sec or not kick_onsets_sec:
            return None

        kick_arr = np.array(kick_onsets_sec)
        beat_interval = 60.0 / bpm

        deviations = []
        for b_onset in bass_onsets_sec:
            # Find nearest kick onset to this bass note
            diffs = np.abs(kick_arr - b_onset)
            nearest_kick = kick_arr[np.argmin(diffs)]
            # Deviation in ms — negative=rushing, positive=dragging
            dev_ms = (b_onset - nearest_kick) * 1000.0
            # Wrap to ±half beat
            half_beat_ms = beat_interval * 500.0
            while dev_ms > half_beat_ms:
                dev_ms -= beat_interval * 1000
            while dev_ms < -half_beat_ms:
                dev_ms += beat_interval * 1000
            deviations.append(dev_ms)

        if not deviations:
            return None

        mean_dev   = float(np.mean(deviations))
        std_dev    = float(np.std(deviations))

        # Pocket score: high when deviation AND std-dev are small
        pocket_score = math.exp(
            -(mean_dev**2) / (2 * self.EXPRESSIVE_MS**2)
        ) * math.exp(
            -(std_dev**2) / (2 * self.EXPRESSIVE_MS**2)
        )

        radar_angle     = np.clip(mean_dev / self.ERRATIC_MS, -1.0, 1.0)
        radar_magnitude = min(1.0, std_dev / self.ERRATIC_MS)

        if abs(mean_dev) <= self.LOCKED_MS and std_dev <= self.LOCKED_MS:
            colour = "green"
        elif abs(mean_dev) <= self.EXPRESSIVE_MS and std_dev <= self.EXPRESSIVE_MS:
            colour = "amber"
        else:
            colour = "red"

        return PocketRadarFrame(
            bassist_deviation_ms   = mean_dev,
            bassist_consistency_ms = std_dev,
            drummer_grid_bpm       = bpm,
            pocket_score           = float(pocket_score),
            radar_angle            = float(radar_angle),
            radar_magnitude        = float(radar_magnitude),
            aura_colour            = colour,
        )


# ─── 2. Topographic Frequency Map (extracted to topographic_map.py) ───────────
# Re-exported for backward compatibility
from .topographic_map import FreqCollisionZone, TopographicFreqMap, TopographicFreqMapFrame

# ─── 3. Performance Coach ─────────────────────────────────────────────────────

@dataclass
class CoachingNote:
    """A single coaching suggestion."""
    domain:     str    # "timing", "dynamics", "tone", "room"
    severity:   str    # "info", "tip", "warning"
    message:    str
    metric:     float  # the measured value that triggered this note
    threshold:  float  # the threshold that was crossed


class PerformanceCoach:
    """
    Analyses a session capture and emits human-readable coaching notes.

    Currently covers:
      - Timing: rushing/dragging tendencies, consistency issues
      - Dynamics: dynamic range compression, velocity monotony
      - Tone: harsh pick attack (transient spike detection proxy)
    """

    def analyse(
        self,
        groove_lean_ms: float,
        groove_consistency_ms: float,
        velocity_range_ratio: float,
        pick_attack_ratio: float = 0.0,   # 0–1; 1 = extremely harsh attack
    ) -> list[CoachingNote]:
        notes: list[CoachingNote] = []

        # Timing: rushing
        if groove_lean_ms < -10.0:
            notes.append(CoachingNote(
                domain="timing", severity="tip",
                message=(
                    f"You are rushing the beat by {abs(groove_lean_ms):.1f} ms on average. "
                    "Try relaxing your wrist and letting the downbeat come to you — "
                    "the pocket is just behind the click."
                ),
                metric=groove_lean_ms, threshold=-10.0,
            ))

        # Timing: dragging
        if groove_lean_ms > 15.0:
            notes.append(CoachingNote(
                domain="timing", severity="tip",
                message=(
                    f"Your groove is sitting {groove_lean_ms:.1f} ms behind the grid. "
                    "This can feel intentionally heavy — confirm it is deliberate "
                    "or try leading with your pick/finger slightly earlier."
                ),
                metric=groove_lean_ms, threshold=15.0,
            ))

        # Timing: inconsistency
        if groove_consistency_ms > 12.0:
            notes.append(CoachingNote(
                domain="timing", severity="warning",
                message=(
                    f"Your timing consistency is {groove_consistency_ms:.1f} ms std-dev. "
                    "High variance suggests fatigue or concentration drift. "
                    "Take a short break and reset before the next take."
                ),
                metric=groove_consistency_ms, threshold=12.0,
            ))

        # Dynamics: monotony
        if velocity_range_ratio < 0.15:
            notes.append(CoachingNote(
                domain="dynamics", severity="info",
                message=(
                    "Your velocity range is very narrow this take. "
                    "Try exaggerating the natural strong/weak beats — "
                    "let your accents speak and your ghost notes whisper."
                ),
                metric=velocity_range_ratio, threshold=0.15,
            ))

        # Tone: harsh pick attack
        if pick_attack_ratio > 0.7:
            notes.append(CoachingNote(
                domain="tone", severity="tip",
                message=(
                    "High transient energy detected in the 3–6 kHz range — "
                    "this usually indicates a sharp pick angle hitting the string. "
                    "Try rotating your pick 10–15 degrees and softening your wrist. "
                    "You will retain attack clarity while losing the scrape."
                ),
                metric=pick_attack_ratio, threshold=0.7,
            ))

        return notes
