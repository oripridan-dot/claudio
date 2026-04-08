"""
claudio_server.py — FastAPI WebSocket Bridge

Real-time bidirectional communication between Claudio's Python AI engine
and the React frontend. Streams:
  - Instrument detections → UI
  - Phase correlation frames → UI
  - Room scan results → UI
  - Sweet spot corrections → UI
  - Mentor tips → UI
  - Roadmap state → UI
  - Camera telemetry ← UI
  - User actions ← UI

Zero-latency: AI analysis runs asynchronously on the observation path.
The live audio path is never touched by this server.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from claudio.intelligence.instrument_classifier import InstrumentClassifier
from claudio.intelligence.multimodal_fusion import (
    BoundingBox,
    MultimodalFusion,
    VisionDetection,
    VisualCategory,
)
from claudio.intelligence.phase_detector import PhaseCorrelationMeter
from claudio.intelligence.room_scanner import RoomScanner
from claudio.intelligence.sweet_spot_engine import (
    ListenerPosition,
    SweetSpotEngine,
)
from claudio.mentor.knowledge_base import (
    MentorKnowledgeBase,
    TriggerCategory,
)
from claudio.mentor.roadmap_engine import RoadmapEngine
from claudio.metering.semantic_metering import (
    AcousticEnvironmentAdvisor,
    PerformanceCoach,
    PocketRadar,
    TopographicFreqMap,
)

app = FastAPI(title="Claudio Intelligence Server", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Shared Engine Instances ──────────────────────────────────────────────────

classifier = InstrumentClassifier(sample_rate=48_000)
fusion = MultimodalFusion()
phase_meter = PhaseCorrelationMeter(sample_rate=48_000)
room_scanner = RoomScanner(sample_rate=48_000)
sweet_spot = SweetSpotEngine()
knowledge_base = MentorKnowledgeBase()
roadmap = RoadmapEngine()
pocket_radar = PocketRadar()
freq_map = TopographicFreqMap(sample_rate=48_000)
coach = PerformanceCoach()
advisor = AcousticEnvironmentAdvisor()


def _serialize(obj: Any) -> Any:
    """Convert dataclasses/numpy to JSON-safe dicts."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dataclass_fields__"):
        d = {}
        for k, v in asdict(obj).items():
            d[k] = _serialize(v)
        return d
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "value"):
        return obj.value
    return obj


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "0.3.0",
        "modules": {
            "instrument_classifier": True,
            "multimodal_fusion": True,
            "phase_detector": True,
            "room_scanner": True,
            "sweet_spot_engine": True,
            "knowledge_base": len(knowledge_base.all_tips),
            "roadmap": roadmap.state.current_phase.value,
        },
    }


@app.get("/api/mentors")
async def get_mentors() -> list[dict]:
    return [_serialize(m) for m in knowledge_base.all_mentors]


@app.get("/api/tips")
async def get_tips() -> list[dict]:
    return [_serialize(t) for t in knowledge_base.all_tips]


@app.get("/api/roadmap")
async def get_roadmap() -> dict:
    return _serialize(roadmap.state)


@app.post("/api/roadmap/advance")
async def advance_phase() -> dict:
    new = roadmap.advance_phase()
    return {"advanced_to": new.value if new else None, "state": _serialize(roadmap.state)}


@app.post("/api/roadmap/complete/{item_id}")
async def complete_item(item_id: str) -> dict:
    ok = roadmap.complete_item(item_id)
    return {"completed": ok, "state": _serialize(roadmap.state)}


# ─── WebSocket ───────────────────────────────────────────────────────────────

class SessionState:
    """Per-connection analysis state."""

    def __init__(self) -> None:
        self.instrument_classifier = InstrumentClassifier(sample_rate=48_000)
        self.phase_meter = PhaseCorrelationMeter(sample_rate=48_000)
        self.room_scanner = RoomScanner(sample_rate=48_000)
        self.sweet_spot = SweetSpotEngine()
        self.fusion = MultimodalFusion()
        self.roadmap = RoadmapEngine()
        self.last_instrument: dict = {}
        self.last_phase: dict = {}


@app.websocket("/ws/session")
async def session_ws(ws: WebSocket) -> None:
    await ws.accept()
    session = SessionState()

    # Send initial state
    await ws.send_json({
        "type": "init",
        "roadmap": _serialize(session.roadmap.state),
        "mentors": [_serialize(m) for m in knowledge_base.all_mentors],
    })

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "audio_buffer":
                # Expects: { type: "audio_buffer", samples: [...], channel_id: "..." }
                samples = np.array(data["samples"], dtype=np.float32)

                # Instrument classification
                detection = session.instrument_classifier.classify(samples)
                result = _serialize(detection)
                await ws.send_json({"type": "instrument_detection", "data": result})

                # Check for coaching triggers
                _check_coaching_triggers(session, detection, ws)

            elif msg_type == "phase_check":
                # Expects: { type: "phase_check", ch1: [...], ch2: [...], ch1_id, ch2_id }
                ch1 = np.array(data["ch1"], dtype=np.float32)
                ch2 = np.array(data["ch2"], dtype=np.float32)
                frame = session.phase_meter.measure(ch1, ch2)
                result = _serialize(frame)
                await ws.send_json({"type": "phase_frame", "data": result})

                # Auto-trigger mentor tip for phase issues
                if frame.needs_polarity_flip:
                    tip = knowledge_base.find_best_tip(
                        TriggerCategory.PHASE_CANCELLATION,
                        confidence=abs(frame.correlation),
                    )
                    if tip:
                        await ws.send_json({
                            "type": "mentor_tip",
                            "data": _serialize(tip),
                        })

            elif msg_type == "room_scan_clap":
                # Expects: { type: "room_scan_clap", audio: [...] }
                audio = np.array(data["audio"], dtype=np.float32)
                scan = session.room_scanner.scan_from_clap(audio)
                result = _serialize(scan)
                await ws.send_json({"type": "room_scan_result", "data": result})

                # Process roadmap detection
                session.roadmap.process_detection("room_scan_complete")
                await ws.send_json({
                    "type": "roadmap_update",
                    "data": _serialize(session.roadmap.state),
                })

                # Trigger acoustic advice
                for adv_item in scan.acoustic_advice:
                    trigger = _advice_to_trigger(adv_item.category)
                    if trigger:
                        tip = knowledge_base.find_best_tip(trigger, confidence=0.7)
                        if tip:
                            await ws.send_json({
                                "type": "mentor_tip",
                                "data": _serialize(tip),
                            })

            elif msg_type == "vision_detection":
                # Expects: { type: "vision_detection", detections: [...] }
                vision_list = []
                for vd in data.get("detections", []):
                    vision_list.append(VisionDetection(
                        category=VisualCategory(vd.get("category", "unknown")),
                        confidence=vd.get("confidence", 0.0),
                        bounding_box=BoundingBox(**vd.get("bounding_box", {
                            "x": 0, "y": 0, "width": 0, "height": 0,
                        })),
                        brand_text=vd.get("brand_text", ""),
                        model_text=vd.get("model_text", ""),
                        body_shape=vd.get("body_shape", ""),
                    ))
                await ws.send_json({
                    "type": "vision_ack",
                    "count": len(vision_list),
                })

            elif msg_type == "head_position":
                # Expects: { type: "head_position", x, y, z }
                pos = ListenerPosition(
                    x=data.get("x", 0.0),
                    y=data.get("y", 0.0),
                    z=data.get("z", 0.0),
                )
                correction = session.sweet_spot.compute_correction(pos)
                await ws.send_json({
                    "type": "sweet_spot_correction",
                    "data": _serialize(correction),
                })

            elif msg_type == "roadmap_action":
                action = data.get("action", "")
                if action == "advance":
                    session.roadmap.advance_phase()
                elif action == "complete":
                    session.roadmap.complete_item(data.get("item_id", ""))
                await ws.send_json({
                    "type": "roadmap_update",
                    "data": _serialize(session.roadmap.state),
                })

            elif msg_type == "ping":
                await ws.send_json({"type": "pong", "ts": time.time()})

    except WebSocketDisconnect:
        pass


async def _check_coaching_triggers(
    session: SessionState,
    detection: Any,
    ws: WebSocket,
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
                        "data": _serialize(tip),
                    })


def _advice_to_trigger(category: str) -> TriggerCategory | None:
    mapping = {
        "bass_buildup": TriggerCategory.BASS_BUILDUP,
        "flutter_echo": TriggerCategory.FLUTTER_ECHO,
        "comb_filter": TriggerCategory.ROOM_REFLECTION,
        "reflection": TriggerCategory.ROOM_REFLECTION,
    }
    return mapping.get(category)
