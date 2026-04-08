"""
room_scanner.py — Acoustic Room Scanning & Analysis Pipeline

Full room characterization from microphone measurement:
  - RT60 estimation (reverb tail decay time)
  - Room mode detection (resonant standing waves)
  - Early reflection analysis (image-source method)
  - Flutter echo detection (comb pattern analysis)
  - Room geometry estimation from impulse response

Uses the hardware_neutralization InverseFIR pipeline for sweep-based
measurement, and adds diagnostic visualization data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RoomMode:
    """A detected standing wave resonance in the room."""
    frequency_hz: float
    magnitude_db: float
    q_factor: float           # sharpness of the resonance peak
    likely_dimension: str     # "length", "width", "height" (estimated)
    treatment_advice: str


@dataclass
class EarlyReflection:
    """A detected early reflection from the impulse response."""
    delay_ms: float           # time after direct sound
    level_db: float           # relative to direct sound
    surface: str              # estimated reflecting surface
    azimuth_deg: float        # estimated direction (0=front, 90=right)


@dataclass
class RoomScanResult:
    """Complete room acoustic analysis result."""
    # Reverb characteristics
    rt60_ms: float              # time for reverb to decay 60dB
    rt60_category: str          # "dead" (<300ms), "controlled" (300-600), "live" (600-1200), "reverberant" (>1200)

    # Standing wave analysis
    room_modes: list[RoomMode]
    worst_mode_hz: float        # most problematic frequency

    # Early reflections
    early_reflections: list[EarlyReflection]
    itdg_ms: float              # Initial Time Delay Gap (direct→first reflection)

    # Flutter echo
    has_flutter_echo: bool
    flutter_frequency_hz: float

    # Room quality score
    acoustic_quality_score: float  # 0-100
    noise_floor_db: float

    # Treatment recommendations
    treatment_plan: list[str]

    # Visualization data
    impulse_response: Optional[np.ndarray] = None
    frequency_response_db: Optional[np.ndarray] = None
    freq_axis_hz: Optional[np.ndarray] = None


class RoomScanner:
    """
    Analyses a room's acoustic characteristics from an impulse response.

    The impulse response can be obtained by:
    1. Clap detection (rough but fast)
    2. Log-swept sine (Farina method) via InverseFIR profiler (accurate)
    3. Balloon pop or starter pistol (field measurement)

    The scanner extracts RT60, room modes, early reflections, flutter echo,
    and generates a comprehensive treatment plan.
    """

    def __init__(self, sample_rate: int = 48_000):
        self._sr = sample_rate

    def scan_from_impulse(self, impulse_response: np.ndarray) -> RoomScanResult:
        """Full room analysis from a measured impulse response."""
        ir = impulse_response.copy()
        if np.max(np.abs(ir)) > 0:
            ir = ir / np.max(np.abs(ir))  # normalize

        rt60 = self._estimate_rt60(ir)
        modes = self._detect_room_modes(ir)
        reflections = self._detect_early_reflections(ir)
        flutter, flutter_freq = self._detect_flutter_echo(ir)
        noise_floor = self._estimate_noise_floor(ir)
        quality = self._compute_quality_score(rt60, modes, flutter, noise_floor)
        treatment = self._generate_treatment_plan(rt60, modes, flutter, reflections)

        # Frequency response for visualization
        n_fft = max(2048, len(ir))
        spectrum = np.abs(np.fft.rfft(ir, n=n_fft))
        freq_response_db = 20 * np.log10(spectrum + 1e-10)
        freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / self._sr)

        # RT60 category
        if rt60 < 300:
            rt60_cat = "dead"
        elif rt60 < 600:
            rt60_cat = "controlled"
        elif rt60 < 1200:
            rt60_cat = "live"
        else:
            rt60_cat = "reverberant"

        worst_mode_hz = modes[0].frequency_hz if modes else 0.0

        # ITDG
        itdg = reflections[0].delay_ms if reflections else 0.0

        return RoomScanResult(
            rt60_ms=rt60,
            rt60_category=rt60_cat,
            room_modes=modes,
            worst_mode_hz=worst_mode_hz,
            early_reflections=reflections,
            itdg_ms=itdg,
            has_flutter_echo=flutter,
            flutter_frequency_hz=flutter_freq,
            acoustic_quality_score=quality,
            noise_floor_db=noise_floor,
            treatment_plan=treatment,
            impulse_response=ir,
            frequency_response_db=freq_response_db.astype(np.float32),
            freq_axis_hz=freq_axis.astype(np.float32),
        )

    def scan_from_clap(self, audio: np.ndarray) -> RoomScanResult:
        """Extract an approximate IR from a hand-clap recording."""
        ir = self._extract_clap_ir(audio)
        return self.scan_from_impulse(ir)

    # ── RT60 Estimation ──────────────────────────────────────────────────

    def _estimate_rt60(self, ir: np.ndarray) -> float:
        """Schroeder backward integration method for RT60."""
        energy = ir ** 2
        # Backward integration (Schroeder curve)
        schroeder = np.cumsum(energy[::-1])[::-1]
        schroeder_db = 10 * np.log10(schroeder / (schroeder[0] + 1e-10) + 1e-10)

        # Find -5dB and -25dB points (T20 extrapolation to T60)
        db_5 = -5.0
        db_25 = -25.0
        idx_5 = None
        idx_25 = None
        for i in range(len(schroeder_db)):
            if schroeder_db[i] <= db_5 and idx_5 is None:
                idx_5 = i
            if schroeder_db[i] <= db_25 and idx_25 is None:
                idx_25 = i
                break

        if idx_5 is not None and idx_25 is not None and idx_25 > idx_5:
            t20_ms = ((idx_25 - idx_5) / self._sr) * 1000
            rt60 = t20_ms * 3.0  # extrapolate T20 → T60
        else:
            # Fallback: simple -60dB point
            for i in range(len(schroeder_db)):
                if schroeder_db[i] <= -60:
                    return (i / self._sr) * 1000
            rt60 = (len(ir) / self._sr) * 1000

        return rt60

    # ── Room Mode Detection ──────────────────────────────────────────────

    def _detect_room_modes(self, ir: np.ndarray) -> list[RoomMode]:
        """Detect resonant peaks in the low-frequency spectrum."""
        n_fft = max(4096, len(ir))
        spectrum = np.abs(np.fft.rfft(ir, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self._sr)
        magnitude_db = 20 * np.log10(spectrum + 1e-10)

        # Focus on 20-500 Hz (room modes range)
        mode_mask = (freqs >= 20) & (freqs <= 500)
        mode_freqs = freqs[mode_mask]
        mode_mag = magnitude_db[mode_mask]

        if len(mode_mag) == 0:
            return []

        # Find peaks: local maxima above mean + 1 std
        threshold = float(np.mean(mode_mag) + np.std(mode_mag))
        modes: list[RoomMode] = []

        for i in range(1, len(mode_mag) - 1):
            if (mode_mag[i] > mode_mag[i - 1]
                    and mode_mag[i] > mode_mag[i + 1]
                    and mode_mag[i] > threshold):
                freq = float(mode_freqs[i])
                mag = float(mode_mag[i])

                # Q factor estimation (3dB bandwidth)
                half_power = mag - 3.0
                left_idx = i
                right_idx = i
                while left_idx > 0 and mode_mag[left_idx] > half_power:
                    left_idx -= 1
                while right_idx < len(mode_mag) - 1 and mode_mag[right_idx] > half_power:
                    right_idx += 1
                bandwidth = float(mode_freqs[right_idx] - mode_freqs[left_idx]) + 0.1
                q = freq / bandwidth

                # Estimate which dimension
                speed_of_sound = 343.0
                half_wavelength = speed_of_sound / (2 * freq)
                if half_wavelength > 3.5:
                    dim = "length"
                elif half_wavelength > 2.0:
                    dim = "width"
                else:
                    dim = "height"

                advice = (
                    f"Room mode at {freq:.0f}Hz (λ/2 ≈ {half_wavelength:.1f}m, "
                    f"likely {dim}). Place a bass trap at the boundary wall "
                    f"perpendicular to the room's {dim}."
                )

                modes.append(RoomMode(
                    frequency_hz=freq,
                    magnitude_db=mag,
                    q_factor=q,
                    likely_dimension=dim,
                    treatment_advice=advice,
                ))

        # Sort by magnitude (worst first)
        modes.sort(key=lambda m: m.magnitude_db, reverse=True)
        return modes[:10]  # top 10 modes

    # ── Early Reflection Detection ───────────────────────────────────────

    def _detect_early_reflections(self, ir: np.ndarray) -> list[EarlyReflection]:
        """Detect early reflections from IR peak analysis."""
        envelope = np.abs(ir)
        # Find direct sound peak
        direct_idx = int(np.argmax(envelope[:int(self._sr * 0.01)]))  # within first 10ms
        direct_level = envelope[direct_idx]

        if direct_level < 1e-8:
            return []

        reflections: list[EarlyReflection] = []
        # Search 5ms to 80ms after direct sound
        start = direct_idx + int(self._sr * 0.005)
        end = min(len(envelope), direct_idx + int(self._sr * 0.080))

        # Find peaks in the envelope
        threshold = direct_level * 0.1  # -20dB relative to direct
        for i in range(start + 1, end - 1):
            if (envelope[i] > envelope[i - 1]
                    and envelope[i] > envelope[i + 1]
                    and envelope[i] > threshold):
                delay_ms = ((i - direct_idx) / self._sr) * 1000
                level_db = 20 * math.log10(envelope[i] / (direct_level + 1e-10))

                # Estimate surface from delay
                distance_m = delay_ms * 0.343 / 2  # round-trip
                if distance_m < 1.0:
                    surface = "desk_reflection"
                    azimuth = 0.0
                elif distance_m < 2.0:
                    surface = "side_wall"
                    azimuth = 90.0 if len(reflections) % 2 == 0 else -90.0
                elif distance_m < 3.0:
                    surface = "floor_or_ceiling"
                    azimuth = 0.0
                else:
                    surface = "back_wall"
                    azimuth = 180.0

                reflections.append(EarlyReflection(
                    delay_ms=delay_ms,
                    level_db=level_db,
                    surface=surface,
                    azimuth_deg=azimuth,
                ))

        # Sort by delay
        reflections.sort(key=lambda r: r.delay_ms)
        return reflections[:8]  # top 8

    # ── Flutter Echo Detection ───────────────────────────────────────────

    def _detect_flutter_echo(self, ir: np.ndarray) -> tuple[bool, float]:
        """Detect flutter echo (regular comb pattern in the spectrum)."""
        n_fft = max(4096, len(ir))
        spectrum = np.abs(np.fft.rfft(ir, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self._sr)

        # Focus on 500-5000 Hz (flutter echo audible range)
        mask = (freqs >= 500) & (freqs <= 5000)
        s = spectrum[mask]
        f = freqs[mask]

        if len(s) < 10:
            return False, 0.0

        # Find peaks
        threshold = float(np.mean(s) + np.std(s))
        peak_freqs: list[float] = []
        for i in range(1, len(s) - 1):
            if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > threshold:
                peak_freqs.append(float(f[i]))

        if len(peak_freqs) < 3:
            return False, 0.0

        # Check for regular spacing (hallmark of flutter echo)
        intervals = np.diff(peak_freqs)
        if len(intervals) < 2:
            return False, 0.0

        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))
        cv = std_interval / (mean_interval + 1e-6)  # coefficient of variation

        if cv < 0.15:  # highly regular spacing = flutter echo
            return True, mean_interval

        return False, 0.0

    # ── Noise Floor ──────────────────────────────────────────────────────

    def _estimate_noise_floor(self, ir: np.ndarray) -> float:
        """Estimate noise floor from the tail of the IR."""
        tail_start = int(len(ir) * 0.8)
        if tail_start >= len(ir):
            return -96.0
        tail = ir[tail_start:]
        rms = float(np.sqrt(np.mean(tail ** 2))) + 1e-10
        return 20 * math.log10(rms)

    # ── Quality Score ────────────────────────────────────────────────────

    def _compute_quality_score(
        self,
        rt60: float,
        modes: list[RoomMode],
        flutter: bool,
        noise_floor_db: float,
    ) -> float:
        """0-100 acoustic quality score for the room."""
        score = 100.0

        # RT60: ideal for recording is 300-600ms
        if rt60 < 200:
            score -= 15  # too dead
        elif rt60 > 800:
            score -= min(30, (rt60 - 800) / 50)

        # Room modes penalty
        score -= min(30, len(modes) * 5)

        # Flutter echo is bad
        if flutter:
            score -= 20

        # Noise floor: should be below -60dB
        if noise_floor_db > -40:
            score -= 20
        elif noise_floor_db > -55:
            score -= 10

        return max(0.0, min(100.0, score))

    # ── Treatment Plan ───────────────────────────────────────────────────

    def _generate_treatment_plan(
        self,
        rt60: float,
        modes: list[RoomMode],
        flutter: bool,
        reflections: list[EarlyReflection],
    ) -> list[str]:
        plan: list[str] = []

        if rt60 > 800:
            plan.append(
                f"RT60 is {rt60:.0f}ms — too reverberant for tracking. "
                f"Add absorption panels on the first reflection points "
                f"(side walls at ear height) and behind the listening position."
            )
        elif rt60 < 200:
            plan.append(
                f"RT60 is {rt60:.0f}ms — room is acoustically dead. "
                f"Remove some absorption and add diffusers to bring life back. "
                f"A room that's too dead feels claustrophobic to perform in."
            )

        if flutter:
            plan.append(
                "Flutter echo detected between parallel surfaces. "
                "Break up the parallel geometry: angle one reflective surface, "
                "add a bookshelf, or apply diffusion treatment to one wall."
            )

        for mode in modes[:3]:
            plan.append(mode.treatment_advice)

        # First reflection treatment
        strong_reflections = [r for r in reflections if r.level_db > -10]
        if strong_reflections:
            plan.append(
                f"Strong early reflections detected ({len(strong_reflections)} "
                f"within -10dB of direct sound). Place 2-inch acoustic panels "
                f"at the mirror points on side walls and ceiling."
            )

        if not plan:
            plan.append(
                "Room acoustics are in good shape! Minor tuning with "
                "strategic diffuser placement can further improve imaging."
            )

        return plan

    # ── Clap IR Extraction ───────────────────────────────────────────────

    def _extract_clap_ir(self, audio: np.ndarray) -> np.ndarray:
        """Extract an approximate impulse response from a clap recording."""
        envelope = np.abs(audio)
        # Find the clap (loudest transient)
        peak_idx = int(np.argmax(envelope))
        # Take 1 second after the peak as the IR
        ir_length = min(self._sr, len(audio) - peak_idx)
        ir = audio[peak_idx:peak_idx + ir_length].copy()
        return ir
