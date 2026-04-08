"""
knowledge_base.py — Pro-Tip Mentorship Knowledge Base

Structured database of verified acoustic & engineering principles,
grounded in real-world testimonials from top-tier studio engineers.

Each MentorTip is:
  - Triggered by a specific acoustic/visual detection event
  - Attributed to a real engineer with photo, studio, and date
  - Contains a concrete physical action (not a digital preset)
  - Categorized by production phase (setup/tracking/mixing/mastering)

The UI renders these as "Mentorship Moment" cards — a frosted-glass overlay
with the engineer's photo, their quote, and an interactive action guide.

The actual tip data lives in mentor_tips.py (separated for 300-line compliance).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProductionPhase(Enum):
    PRE_PRODUCTION = "pre_production"
    SETUP = "setup"
    TRACKING = "tracking"
    MIXING = "mixing"
    MASTERING = "mastering"


class TriggerCategory(Enum):
    PHASE_CANCELLATION = "phase_cancellation"
    PROXIMITY_EFFECT = "proximity_effect"
    ROOM_REFLECTION = "room_reflection"
    FLUTTER_ECHO = "flutter_echo"
    BASS_BUILDUP = "bass_buildup"
    HARSH_TRANSIENT = "harsh_transient"
    GAIN_STAGING = "gain_staging"
    MIC_PLACEMENT = "mic_placement"
    SPEAKER_PLACEMENT = "speaker_placement"
    DYNAMIC_RANGE = "dynamic_range"
    FREQUENCY_MASKING = "frequency_masking"
    TIMING_DRIFT = "timing_drift"
    ENERGY_DROP = "energy_drop"
    LOUDNESS_WAR = "loudness_war"
    MONO_COMPATIBILITY = "mono_compatibility"
    MIC_TECHNIQUE = "mic_technique"
    INSTRUMENT_SETUP = "instrument_setup"
    ROOM_TREATMENT = "room_treatment"


@dataclass
class MentorProfile:
    """A real-world audio engineer / producer."""
    name: str
    title: str                    # e.g. "Grammy-winning Recording Engineer"
    photo_asset: str              # path to portrait image asset
    notable_works: list[str]      # e.g. ["Michael Jackson - Thriller", "Quincy Jones"]
    era: str                      # e.g. "1980s-present"
    specialty: str                # e.g. "Vocal recording & orchestral mixing"


@dataclass
class MentorTip:
    """A single mentorship moment — triggered by detection, grounded in expertise."""
    tip_id: str
    trigger: TriggerCategory
    phase: ProductionPhase
    mentor: MentorProfile
    quote: str                    # the actual wisdom (their words)
    context_location: str         # where the wisdom was shared
    context_date: str             # when
    physical_action: str          # concrete physical correction for the user
    ui_action: str                # what the UI should do ("HIGHLIGHT_PHASE_BUTTON", "SHOW_MIC_ARROW")
    severity: str                 # "info", "tip", "warning", "critical"
    confidence_threshold: float   # minimum detection confidence to trigger this tip
    related_detection: str = ""   # e.g. "DRUM_SNARE_PHASE_180", "VOCAL_PROXIMITY_03_INCH"


# ─── Tip Retrieval Engine ────────────────────────────────────────────────────

def _load_tips() -> list[MentorTip]:
    """Lazy import to break circular dependency with mentor_tips.py."""
    from .mentor_tips import MENTOR_TIPS
    return MENTOR_TIPS


class MentorKnowledgeBase:
    """
    Retrieves the most relevant mentorship tip for a given detection event.
    """

    def __init__(self) -> None:
        tips = _load_tips()
        self._tips = {tip.tip_id: tip for tip in tips}
        self._by_trigger: dict[TriggerCategory, list[MentorTip]] = {}
        for tip in tips:
            self._by_trigger.setdefault(tip.trigger, []).append(tip)
        self._all_tips = tips

    def get_tip(self, tip_id: str) -> MentorTip | None:
        return self._tips.get(tip_id)

    def find_tips(
        self,
        trigger: TriggerCategory,
        phase: ProductionPhase | None = None,
        confidence: float = 0.0,
    ) -> list[MentorTip]:
        """Find all tips matching a trigger category, filtered by phase and confidence."""
        candidates = self._by_trigger.get(trigger, [])
        results = []
        for tip in candidates:
            if tip.confidence_threshold > confidence:
                continue
            if phase is not None and tip.phase != phase:
                continue
            results.append(tip)
        return results

    def find_best_tip(
        self,
        trigger: TriggerCategory,
        phase: ProductionPhase | None = None,
        confidence: float = 0.0,
    ) -> MentorTip | None:
        """Return the single most relevant tip for the detection."""
        tips = self.find_tips(trigger, phase, confidence)
        if not tips:
            # Try without phase filter
            tips = self.find_tips(trigger, confidence=confidence)
        return tips[0] if tips else None

    @property
    def all_tips(self) -> list[MentorTip]:
        return self._all_tips

    @property
    def all_mentors(self) -> list[MentorProfile]:
        seen: dict[str, MentorProfile] = {}
        for tip in self._all_tips:
            if tip.mentor.name not in seen:
                seen[tip.mentor.name] = tip.mentor
        return list(seen.values())
