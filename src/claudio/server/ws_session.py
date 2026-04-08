"""
ws_session.py — WebSocket Session State and Handler Helpers

Per-connection state and message processing helpers for the Claudio
intelligence server's /ws/session endpoint.
Extracted from claudio_server.py for 300-line compliance.
"""
from __future__ import annotations

from typing import Any

from claudio.intelligence.instrument_classifier import InstrumentClassifier
from claudio.intelligence.multimodal_fusion import MultimodalFusion
from claudio.intelligence.phase_detector import PhaseCorrelationMeter
from claudio.intelligence.room_scanner import RoomScanner
from claudio.intelligence.sweet_spot_engine import SweetSpotEngine
from claudio.mentor.knowledge_base import MentorKnowledgeBase, TriggerCategory
from claudio.mentor.roadmap_engine import RoadmapEngine


class SessionState:
    """Per-connection analysis state."""

    def __init__(self) -> None:
        self.instrument_classifier = InstrumentClassifier(sample_rate=48_000)
        self.phase_meter = PhaseCorrelationMeter(sample_rate=48_000)
        self.room_scanner = RoomScanner(sample_rate=48_000)
        self.sweet_spot = SweetSpotEngine()
        self.sweet_spot.set_stereo_pair()  # initialize default stereo monitoring
        self.fusion = MultimodalFusion()
        self.roadmap = RoadmapEngine()
        self.last_instrument: dict = {}
        self.last_phase: dict = {}


async def check_coaching_triggers(
    session: SessionState,
    detection: Any,
    ws: Any,
    knowledge_base: MentorKnowledgeBase,
    serialize_fn: Any,
) -> None:
    """Check if the instrument detection triggers any mentor tips."""
    if hasattr(detection, "coaching_hints") and detection.coaching_hints:
        for hint in detection.coaching_hints:
            if "harsh" in hint.lower() or "pick" in hint.lower():
                tip = knowledge_base.find_best_tip(
                    TriggerCategory.HARSH_TRANSIENT,
                    confidence=0.7,
                )
                if tip:
                    await ws.send_json({
                        "type": "mentor_tip",
                        "data": serialize_fn(tip),
                    })


def treatment_text_to_trigger(text: str) -> TriggerCategory | None:
    """Match treatment plan text to a mentor trigger category via keywords."""
    lower = text.lower()
    if "bass trap" in lower or "low-frequency" in lower or "room mode" in lower:
        return TriggerCategory.BASS_BUILDUP
    if "flutter" in lower:
        return TriggerCategory.FLUTTER_ECHO
    if "reflection" in lower or "mirror point" in lower or "comb" in lower:
        return TriggerCategory.ROOM_REFLECTION
    return None
