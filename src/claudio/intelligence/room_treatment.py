"""
room_treatment.py — Room Treatment Plan Generation

Generates actionable acoustic treatment recommendations based on
measured room characteristics: RT60, room modes, flutter echo, reflections.

Extracted from room_scanner.py for single-responsibility compliance.
"""
from __future__ import annotations

from .room_scanner import EarlyReflection, RoomMode


def generate_treatment_plan(
    rt60: float,
    modes: list[RoomMode],
    flutter: bool,
    reflections: list[EarlyReflection],
) -> list[str]:
    """Generate actionable treatment recommendations."""
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


def compute_quality_score(
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
