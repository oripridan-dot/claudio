"""
semantic_metering.py — Semantic Metering Engine

Replaces traditional dBFS/LUFS meters with performance-aware meters that
visualise musical intent, not electrical voltage.

Meters:
  1. PocketRadar       — drummer/bassist groove pocket (ITD between players)
  2. TopographicFreqMap — 3D frequency collision map (mud detection)
  3. PerformanceCoach  — real-time micro-timing and pick-attack guidance
  4. AcousticAdvisor   — room correction hints from mic analysis

All meters are stateless processors — they receive audio/MIDI buffers and
return visualisation-ready data structures.  No side effects.
The UI (claudio-studio-app) consumes these data structures to render the
real-time holographic displays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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
    ) -> Optional[PocketRadarFrame]:
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
            while dev_ms >  half_beat_ms: dev_ms -= beat_interval * 1000
            while dev_ms < -half_beat_ms: dev_ms += beat_interval * 1000
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


# ─── 2. Topographic Frequency Map ────────────────────────────────────────────

@dataclass
class FreqCollisionZone:
    """A detected frequency range where two sources are masking each other."""
    freq_hz_low:  float
    freq_hz_high: float
    source_a:     str
    source_b:     str
    collision_db: float   # energy overlap in dB — higher = more mud
    severity:     str     # "low", "medium", "critical"


@dataclass
class TopographicFreqMapFrame:
    """
    3D topographic map data for the frequency collision display.

    UI rendering:
      - X axis: frequency (20 Hz – 20 kHz, log scale)
      - Y axis: signal energy (dB)
      - Z/colour: collision severity (blue=clear, yellow=warning, red=critical)
      - Collision zones are highlighted with a glowing red mesh
    """
    source_spectra:    dict[str, np.ndarray]   # source_id → 1024-bin magnitude spectrum
    collision_zones:   list[FreqCollisionZone]
    total_mud_score:   float  # 0.0 (crystal clear) – 1.0 (heavily masked)
    freq_bins_hz:      np.ndarray              # 1024-element log-frequency axis


class TopographicFreqMap:
    """
    Analyses frequency spectra from all active sources and identifies
    masking collisions — the "mud zones" in the mix.

    Physics:
      - Two sources mask each other when their simultaneous energy in a
        critical band (Bark scale) exceeds the threshold of masking.
      - Simplified model: collision if overlap energy > 6 dB in any 1/3-oct band.
    """

    FFT_SIZE    = 2048
    N_BINS      = 1024
    COLLISION_DB = 6.0   # overlap threshold for "muddy" masking

    def __init__(self, sample_rate: int = 192_000) -> None:
        self._sr = sample_rate
        self._freq_bins = np.geomspace(20, sample_rate / 2, self.N_BINS)

    def compute(
        self,
        source_buffers: dict[str, np.ndarray],  # source_id → mono audio block
    ) -> TopographicFreqMapFrame:
        if not source_buffers:
            return TopographicFreqMapFrame(
                source_spectra={}, collision_zones=[],
                total_mud_score=0.0, freq_bins_hz=self._freq_bins,
            )

        source_spectra: dict[str, np.ndarray] = {}
        for sid, buf in source_buffers.items():
            if len(buf) < self.FFT_SIZE:
                buf = np.pad(buf, (0, self.FFT_SIZE - len(buf)))
            window  = np.hanning(self.FFT_SIZE)
            spectrum = np.abs(np.fft.rfft(buf[:self.FFT_SIZE] * window))
            # Interpolate to N_BINS log-frequency bins
            fft_freqs  = np.fft.rfftfreq(self.FFT_SIZE, d=1.0 / self._sr)
            log_mag    = np.interp(self._freq_bins, fft_freqs, spectrum)
            source_spectra[sid] = 20 * np.log10(log_mag + 1e-10)

        # Detect collision zones
        collision_zones = []
        ids = list(source_spectra.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sid_a = ids[i]
                sid_b = ids[j]
                s_a   = source_spectra[sid_a]
                s_b   = source_spectra[sid_b]
                overlap = np.minimum(s_a, s_b)
                # Collision: both sources > −40 dBFS and overlap > threshold
                active  = (s_a > -40) & (s_b > -40)
                collide = active & (overlap > self.COLLISION_DB)
                if np.any(collide):
                    col_freqs = self._freq_bins[collide]
                    col_db    = float(np.mean(overlap[collide]))
                    severity  = (
                        "critical" if col_db > 18
                        else "medium" if col_db > 10
                        else "low"
                    )
                    collision_zones.append(FreqCollisionZone(
                        freq_hz_low  = float(col_freqs.min()),
                        freq_hz_high = float(col_freqs.max()),
                        source_a     = sid_a,
                        source_b     = sid_b,
                        collision_db = col_db,
                        severity     = severity,
                    ))

        mud_score = min(1.0, len(collision_zones) * 0.15 +
                        sum(z.collision_db / 40.0 for z in collision_zones))

        return TopographicFreqMapFrame(
            source_spectra  = source_spectra,
            collision_zones = collision_zones,
            total_mud_score = float(mud_score),
            freq_bins_hz    = self._freq_bins,
        )


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


# ─── 4. Acoustic Environment Advisor ─────────────────────────────────────────

@dataclass
class AcousticAdvice:
    """A room-treatment or placement suggestion."""
    category:   str   # "flutter_echo", "bass_buildup", "comb_filter", "reflection"
    description: str
    action:     str   # concrete corrective action


class AcousticEnvironmentAdvisor:
    """
    Analyses a mono mic recording and detects acoustic pathologies:
      - Flutter echo: comb-filter peaks with regular spacing in the spectrum
      - Bass buildup: excessive energy below 200 Hz (room mode)
      - Comb filtering: mic too close to a reflective surface

    All analysis runs on a short analysis window (≈1 second of audio).
    """

    def analyse(
        self,
        audio_mono: np.ndarray,
        sample_rate: int = 48_000,
    ) -> list[AcousticAdvice]:
        advice: list[AcousticAdvice] = []

        if len(audio_mono) < sample_rate // 4:
            return advice

        window  = audio_mono[: sample_rate]  # first second
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        freqs    = np.fft.rfftfreq(len(window), d=1.0 / sample_rate)

        # Bass buildup: energy ratio below 200 Hz vs 200–2000 Hz
        bass_mask = freqs < 200
        mid_mask  = (freqs >= 200) & (freqs < 2000)
        if mid_mask.any() and bass_mask.any():
            bass_energy = float(np.mean(spectrum[bass_mask])) + 1e-10
            mid_energy  = float(np.mean(spectrum[mid_mask]))  + 1e-10
            if bass_energy / mid_energy > 3.0:
                advice.append(AcousticAdvice(
                    category="bass_buildup",
                    description=(
                        "Significant low-frequency room mode detected below 200 Hz. "
                        "Your room is reinforcing bass frequencies, causing mud."
                    ),
                    action=(
                        "Move your microphone stand away from room corners by at least 1 m. "
                        "Corner placement maximises room mode excitation. "
                        "Bass traps in the corner behind you will also help dramatically."
                    ),
                ))

        # Flutter echo: look for periodic spectral peaks (comb pattern)
        peak_intervals = self._detect_comb_pattern(spectrum, freqs)
        if peak_intervals and len(peak_intervals) >= 3:
            advice.append(AcousticAdvice(
                category="flutter_echo",
                description=(
                    "Flutter echo pattern detected — likely caused by two parallel "
                    "reflective surfaces (e.g., opposite walls or ceiling/floor)."
                ),
                action=(
                    "Rotate your microphone 45 degrees away from the glass window or "
                    "parallel wall. Hanging a duvet or acoustic panel on the reflective "
                    "surface opposite your mic will neutralise the flutter immediately."
                ),
            ))

        return advice

    @staticmethod
    def _detect_comb_pattern(
        spectrum: np.ndarray, freqs: np.ndarray, n_peaks: int = 8
    ) -> list[float]:
        """Detect regularly-spaced spectral peaks (comb filter signature)."""
        # Find top N peaks
        peak_indices = []
        threshold = np.mean(spectrum) + np.std(spectrum)
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > threshold and spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                peak_indices.append(i)
        if len(peak_indices) < 3:
            return []

        # Check for regular spacing
        peak_freqs = freqs[peak_indices[:n_peaks]]
        if len(peak_freqs) < 3:
            return []
        intervals  = np.diff(peak_freqs)
        if np.std(intervals) / (np.mean(intervals) + 1e-6) < 0.15:
            # Highly regular spacing → comb filter
            return intervals.tolist()
        return []
