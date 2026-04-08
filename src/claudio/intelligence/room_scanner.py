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

Treatment plan generation lives in room_treatment.py (extracted for
single-responsibility compliance).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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
    rt60_ms: float
    rt60_category: str
    room_modes: list[RoomMode]
    worst_mode_hz: float
    early_reflections: list[EarlyReflection]
    itdg_ms: float
    has_flutter_echo: bool
    flutter_frequency_hz: float
    acoustic_quality_score: float
    noise_floor_db: float
    treatment_plan: list[str]
    impulse_response: np.ndarray | None = None
    frequency_response_db: np.ndarray | None = None
    freq_axis_hz: np.ndarray | None = None


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
        from .room_treatment import compute_quality_score, generate_treatment_plan

        ir = impulse_response.copy()
        if np.max(np.abs(ir)) > 0:
            ir = ir / np.max(np.abs(ir))

        rt60 = self._estimate_rt60(ir)
        modes = self._detect_room_modes(ir)
        reflections = self._detect_early_reflections(ir)
        flutter, flutter_freq = self._detect_flutter_echo(ir)
        noise_floor = self._estimate_noise_floor(ir)
        quality = compute_quality_score(rt60, modes, flutter, noise_floor)
        treatment = generate_treatment_plan(rt60, modes, flutter, reflections)

        # Frequency response for visualization
        n_fft = max(2048, len(ir))
        spectrum = np.abs(np.fft.rfft(ir, n=n_fft))
        freq_response_db = 20 * np.log10(spectrum + 1e-10)
        freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / self._sr)

        if rt60 < 300:
            rt60_cat = "dead"
        elif rt60 < 600:
            rt60_cat = "controlled"
        elif rt60 < 1200:
            rt60_cat = "live"
        else:
            rt60_cat = "reverberant"

        return RoomScanResult(
            rt60_ms=rt60,
            rt60_category=rt60_cat,
            room_modes=modes,
            worst_mode_hz=modes[0].frequency_hz if modes else 0.0,
            early_reflections=reflections,
            itdg_ms=reflections[0].delay_ms if reflections else 0.0,
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
        schroeder = np.cumsum(energy[::-1])[::-1]
        schroeder_db = 10 * np.log10(schroeder / (schroeder[0] + 1e-10) + 1e-10)

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
            rt60 = t20_ms * 3.0
        else:
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

        mode_mask = (freqs >= 20) & (freqs <= 500)
        mode_freqs = freqs[mode_mask]
        mode_mag = magnitude_db[mode_mask]

        if len(mode_mag) == 0:
            return []

        threshold = float(np.mean(mode_mag) + np.std(mode_mag))
        modes: list[RoomMode] = []
        speed_of_sound = 343.0

        for i in range(1, len(mode_mag) - 1):
            if (mode_mag[i] > mode_mag[i - 1]
                    and mode_mag[i] > mode_mag[i + 1]
                    and mode_mag[i] > threshold):
                freq = float(mode_freqs[i])
                mag = float(mode_mag[i])

                half_power = mag - 3.0
                left_idx = i
                right_idx = i
                while left_idx > 0 and mode_mag[left_idx] > half_power:
                    left_idx -= 1
                while right_idx < len(mode_mag) - 1 and mode_mag[right_idx] > half_power:
                    right_idx += 1
                bandwidth = float(mode_freqs[right_idx] - mode_freqs[left_idx]) + 0.1
                q = freq / bandwidth

                half_wl = speed_of_sound / (2 * freq)
                dim = "length" if half_wl > 3.5 else "width" if half_wl > 2.0 else "height"

                advice = (
                    f"Room mode at {freq:.0f}Hz (λ/2 ≈ {half_wl:.1f}m, "
                    f"likely {dim}). Place a bass trap at the boundary wall "
                    f"perpendicular to the room's {dim}."
                )

                modes.append(RoomMode(
                    frequency_hz=freq, magnitude_db=mag, q_factor=q,
                    likely_dimension=dim, treatment_advice=advice,
                ))

        modes.sort(key=lambda m: m.magnitude_db, reverse=True)
        return modes[:10]

    # ── Early Reflection Detection ───────────────────────────────────────

    def _detect_early_reflections(self, ir: np.ndarray) -> list[EarlyReflection]:
        """Detect early reflections from IR peak analysis."""
        envelope = np.abs(ir)
        direct_idx = int(np.argmax(envelope[:int(self._sr * 0.01)]))
        direct_level = envelope[direct_idx]

        if direct_level < 1e-8:
            return []

        reflections: list[EarlyReflection] = []
        start = direct_idx + int(self._sr * 0.005)
        end = min(len(envelope), direct_idx + int(self._sr * 0.080))

        threshold = direct_level * 0.1
        for i in range(start + 1, end - 1):
            if (envelope[i] > envelope[i - 1]
                    and envelope[i] > envelope[i + 1]
                    and envelope[i] > threshold):
                delay_ms = ((i - direct_idx) / self._sr) * 1000
                level_db = 20 * math.log10(envelope[i] / (direct_level + 1e-10))
                distance_m = delay_ms * 0.343 / 2

                if distance_m < 1.0:
                    surface, azimuth = "desk_reflection", 0.0
                elif distance_m < 2.0:
                    surface = "side_wall"
                    azimuth = 90.0 if len(reflections) % 2 == 0 else -90.0
                elif distance_m < 3.0:
                    surface, azimuth = "floor_or_ceiling", 0.0
                else:
                    surface, azimuth = "back_wall", 180.0

                reflections.append(EarlyReflection(
                    delay_ms=delay_ms, level_db=level_db,
                    surface=surface, azimuth_deg=azimuth,
                ))

        reflections.sort(key=lambda r: r.delay_ms)
        return reflections[:8]

    # ── Flutter Echo Detection ───────────────────────────────────────────

    def _detect_flutter_echo(self, ir: np.ndarray) -> tuple[bool, float]:
        """Detect flutter echo (regular comb pattern in the spectrum)."""
        n_fft = max(4096, len(ir))
        spectrum = np.abs(np.fft.rfft(ir, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self._sr)

        mask = (freqs >= 500) & (freqs <= 5000)
        s = spectrum[mask]
        f = freqs[mask]

        if len(s) < 10:
            return False, 0.0

        threshold = float(np.mean(s) + np.std(s))
        peak_freqs: list[float] = []
        for i in range(1, len(s) - 1):
            if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > threshold:
                peak_freqs.append(float(f[i]))

        if len(peak_freqs) < 3:
            return False, 0.0

        intervals = np.diff(peak_freqs)
        if len(intervals) < 2:
            return False, 0.0

        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))
        cv = std_interval / (mean_interval + 1e-6)

        if cv < 0.15:
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

    # ── Clap IR Extraction ───────────────────────────────────────────────

    def _extract_clap_ir(self, audio: np.ndarray) -> np.ndarray:
        """Extract an approximate impulse response from a clap recording."""
        envelope = np.abs(audio)
        peak_idx = int(np.argmax(envelope))
        ir_length = min(self._sr, len(audio) - peak_idx)
        ir = audio[peak_idx:peak_idx + ir_length].copy()
        return ir
