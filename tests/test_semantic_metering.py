"""test_semantic_metering.py — Unit tests for the semantic metering engine."""
from __future__ import annotations

import math
import numpy as np
import pytest

from semantic_metering import (
    PocketRadar, TopographicFreqMap, PerformanceCoach,
    AcousticEnvironmentAdvisor, CoachingNote,
)


# ─── PocketRadar ─────────────────────────────────────────────────────────────

def test_pocket_radar_locked_in():
    radar = PocketRadar()
    bpm = 120.0
    # Bassist exactly on the kick every beat
    kicks = [i * 0.5 for i in range(16)]
    bass  = [k + 0.002 for k in kicks]   # 2 ms behind kick = tight pocket
    frame = radar.compute(bass, kicks, bpm)
    assert frame is not None
    assert frame.aura_colour == "green"
    assert frame.pocket_score > 0.8


def test_pocket_radar_erratic():
    radar = PocketRadar()
    bpm = 120.0
    kicks = [i * 0.5 for i in range(16)]
    # Bass all over the place
    rng  = np.random.default_rng(42)
    bass = [k + float(rng.uniform(-0.06, 0.06)) for k in kicks]
    frame = radar.compute(bass, kicks, bpm)
    assert frame is not None
    assert frame.aura_colour == "red"


def test_pocket_radar_empty_returns_none():
    radar = PocketRadar()
    assert radar.compute([], [0.5, 1.0], 120.0) is None
    assert radar.compute([0.5, 1.0], [], 120.0) is None


# ─── TopographicFreqMap ───────────────────────────────────────────────────────

def test_collision_detected_for_identical_signals():
    mapper = TopographicFreqMap(sample_rate=48_000)
    # Two identical sine waves at 200 Hz will fully overlap
    t = np.linspace(0, 1, 2048)
    sig = np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.7
    frame = mapper.compute({"bass": sig, "kick": sig})
    assert len(frame.collision_zones) > 0
    assert frame.total_mud_score > 0.0


def test_no_collision_on_silence():
    mapper = TopographicFreqMap(sample_rate=48_000)
    silence = np.zeros(2048, dtype=np.float32)
    frame = mapper.compute({"a": silence, "b": silence})
    # Both below −40 dBFS threshold → no collision
    assert len(frame.collision_zones) == 0


# ─── PerformanceCoach ─────────────────────────────────────────────────────────

def test_coach_rushing_note():
    coach = PerformanceCoach()
    notes = coach.analyse(
        groove_lean_ms=-15.0,
        groove_consistency_ms=4.0,
        velocity_range_ratio=0.4,
    )
    domains = [n.domain for n in notes]
    assert "timing" in domains
    rushing = next(n for n in notes if n.domain == "timing")
    assert "rushing" in rushing.message.lower()


def test_coach_no_notes_for_perfect_take():
    coach = PerformanceCoach()
    notes = coach.analyse(
        groove_lean_ms=2.0,
        groove_consistency_ms=3.0,
        velocity_range_ratio=0.5,
        pick_attack_ratio=0.1,
    )
    assert len(notes) == 0


def test_coach_harsh_pick_attack():
    coach = PerformanceCoach()
    notes = coach.analyse(
        groove_lean_ms=0.0,
        groove_consistency_ms=2.0,
        velocity_range_ratio=0.5,
        pick_attack_ratio=0.85,
    )
    tone_notes = [n for n in notes if n.domain == "tone"]
    assert len(tone_notes) > 0


# ─── AcousticEnvironmentAdvisor ──────────────────────────────────────────────

def test_advisor_detects_bass_buildup():
    advisor = AcousticEnvironmentAdvisor()
    sr = 48_000
    t  = np.linspace(0, 1, sr)
    # Dominant 80 Hz fundamental (room mode) + weak mid content
    audio = (
        np.sin(2 * np.pi * 80 * t) * 0.9
        + np.sin(2 * np.pi * 1000 * t) * 0.05
    ).astype(np.float32)
    advice = advisor.analyse(audio, sample_rate=sr)
    categories = [a.category for a in advice]
    assert "bass_buildup" in categories


def test_advisor_clean_room_no_advice():
    advisor = AcousticEnvironmentAdvisor()
    sr = 48_000
    # Pink-noise-ish signal with balanced spectrum — no pathologies
    rng   = np.random.default_rng(0)
    audio = rng.standard_normal(sr).astype(np.float32) * 0.1
    advice = advisor.analyse(audio, sample_rate=sr)
    # A white-noise signal should not trigger bass_buildup or flutter
    bass_advice = [a for a in advice if a.category == "bass_buildup"]
    assert len(bass_advice) == 0
