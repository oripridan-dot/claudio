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

import contextlib
import os
import time

from dotenv import load_dotenv

if os.getenv("CLOUD_NATIVE_WORKSPACE", "false").lower() != "true":
    load_dotenv()

import asyncio
from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from claudio.codec.neural_codec import NeuralCodec
from claudio.collab.session_manager import SessionManager
from claudio.collab.webrtc_manager import WebRTCManager
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
from claudio.intent.intent_decoder import IntentDecoder
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
from claudio.server import auth
from claudio.server.billing import billing_manager
from claudio.server.collab_router import handle_collab_ws
from claudio.server.ws_session import (
    SessionState,
    check_coaching_triggers,
    treatment_text_to_trigger,
)

app = FastAPI(title="Claudio Intelligence Server", version="1.2.0")

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
collab_manager = SessionManager()
webrtc_manager = WebRTCManager(collab_manager)

# ─── Audio Codec (Real neural compression — replaces fake SemanticVocoder) ────

try:
    audio_codec = NeuralCodec(bandwidth_kbps=6.0)
    print("[Codec] NeuralCodec (EnCodec) initialized @ 6.0 kbps")
except Exception as e:
    audio_codec = None
    print(f"[Codec] Failed to initialize NeuralCodec: {e}")

# Intent decoder kept for collab metadata path (fallback additive synth)
global_ddsp_decoder = IntentDecoder(sample_rate=44100)


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


class AuthRequest(BaseModel):
    username: str


@app.post("/api/auth/token")
async def generate_token(req: AuthRequest) -> dict:
    if not req.username:
        raise HTTPException(status_code=400, detail="Username required")
    token = auth.create_token(req.username)
    return {"token": token}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "2.0.0",
        "architecture": "hybrid-encodec",
        "modules": {
            "neural_codec": audio_codec is not None,
            "codec_bandwidth_kbps": audio_codec.bandwidth_kbps if audio_codec else 0,
            "instrument_classifier": True,
            "multimodal_fusion": True,
            "phase_detector": True,
            "room_scanner": True,
            "sweet_spot_engine": True,
            "knowledge_base": len(knowledge_base.all_tips),
            "roadmap": roadmap.state.current_phase.value,
        },
    }


# ─── Audio WebSocket: real neural codec (EnCodec) ────────────────────────────


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    """High-fidelity audio round-trip via EnCodec neural codec.

    Protocol (binary frames):
      Client → Server : float32 PCM samples (mono, 48000 Hz)
      Server → Client : EnCodec compressed codes (int16), much smaller

    The NeuralCodec compresses audio to 6 kbps (vs 768 kbps raw PCM)
    with near-transparent perceptual quality.
    """
    if audio_codec is None:
        await ws.close(code=1011, reason="NeuralCodec not available")
        return

    await ws.accept()
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await ws.receive_bytes()
            if not raw:
                continue

            audio_np = np.frombuffer(raw, dtype=np.float32).copy()

            def _encode(audio_np=audio_np):
                frame = audio_codec.encode(audio_np, input_sr=48_000)
                return frame.to_bytes()

            compressed = await loop.run_in_executor(None, _encode)
            await ws.send_bytes(compressed)

    except Exception:
        pass
    finally:
        with contextlib.suppress(Exception):
            await ws.close()


@app.websocket("/ws/audio/decode")
async def ws_audio_decode(ws: WebSocket):
    """Decode EnCodec frames back to PCM audio.

    Protocol (binary frames):
      Client → Server : EnCodec compressed codes (int16)
      Server → Client : float32 PCM samples (mono, 48000 Hz)
    """
    if audio_codec is None:
        await ws.close(code=1011, reason="NeuralCodec not available")
        return

    await ws.accept()
    loop = asyncio.get_event_loop()
    n_codebooks = audio_codec.n_codebooks

    try:
        while True:
            raw = await ws.receive_bytes()
            if not raw:
                continue

            from claudio.codec.neural_codec import CodecFrame

            def _decode(raw=raw, n_codebooks=n_codebooks):
                frame = CodecFrame.from_bytes(raw, n_codebooks)
                return audio_codec.decode(frame, target_sr=48_000)

            audio = await loop.run_in_executor(None, _decode)
            await ws.send_bytes(audio.astype(np.float32).tobytes())

    except Exception:
        pass
    finally:
        with contextlib.suppress(Exception):
            await ws.close()


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


@app.websocket("/ws/session")
async def session_ws(ws: WebSocket) -> None:
    token = ws.query_params.get("token")
    if not token or not auth.verify_token(token):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws.accept()
    session = SessionState()

    # Send initial state
    await ws.send_json(
        {
            "type": "init",
            "roadmap": _serialize(session.roadmap.state),
            "mentors": [_serialize(m) for m in knowledge_base.all_mentors],
        }
    )

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
                frame = session.phase_meter.analyze(
                    ch1,
                    ch2,
                    ch1_name=data.get("ch1_id", "CH1"),
                    ch2_name=data.get("ch2_id", "CH2"),
                )
                result = _serialize(frame)
                await ws.send_json({"type": "phase_frame", "data": result})

                # Auto-trigger mentor tip for phase issues
                if frame.needs_polarity_flip:
                    tip = knowledge_base.find_best_tip(
                        TriggerCategory.PHASE_CANCELLATION,
                        confidence=abs(frame.correlation),
                    )
                    if tip:
                        await ws.send_json(
                            {
                                "type": "mentor_tip",
                                "data": _serialize(tip),
                            }
                        )

            elif msg_type == "room_scan_clap":
                # Expects: { type: "room_scan_clap", audio: [...] }
                audio = np.array(data["audio"], dtype=np.float32)
                scan = session.room_scanner.scan_from_clap(audio)
                result = _serialize(scan)
                await ws.send_json({"type": "room_scan_result", "data": result})

                # Process roadmap detection
                session.roadmap.process_detection("room_scan_complete")
                await ws.send_json(
                    {
                        "type": "roadmap_update",
                        "data": _serialize(session.roadmap.state),
                    }
                )

                # Trigger mentor tips from treatment plan keywords
                for plan_item in scan.treatment_plan:
                    trigger = _treatment_text_to_trigger(plan_item)
                    if trigger:
                        tip = knowledge_base.find_best_tip(trigger, confidence=0.7)
                        if tip:
                            await ws.send_json(
                                {
                                    "type": "mentor_tip",
                                    "data": _serialize(tip),
                                }
                            )

            elif msg_type == "vision_detection":
                # Expects: { type: "vision_detection", detections: [...] }
                vision_list = []
                for vd in data.get("detections", []):
                    vision_list.append(
                        VisionDetection(
                            category=VisualCategory(vd.get("category", "unknown")),
                            confidence=vd.get("confidence", 0.0),
                            bounding_box=BoundingBox(
                                **vd.get(
                                    "bounding_box",
                                    {
                                        "x": 0,
                                        "y": 0,
                                        "width": 0,
                                        "height": 0,
                                    },
                                )
                            ),
                            brand_text=vd.get("brand_text", ""),
                            model_text=vd.get("model_text", ""),
                            body_shape=vd.get("body_shape", ""),
                        )
                    )
                await ws.send_json(
                    {
                        "type": "vision_ack",
                        "count": len(vision_list),
                    }
                )

            elif msg_type == "head_position":
                # Expects: { type: "head_position", x, y, z }
                pos = ListenerPosition(
                    x=data.get("x", 0.0),
                    y=data.get("y", 0.0),
                    z=data.get("z", 0.0),
                )
                correction = session.sweet_spot.compute(pos)
                await ws.send_json(
                    {
                        "type": "sweet_spot_correction",
                        "data": _serialize(correction),
                    }
                )

            elif msg_type == "roadmap_action":
                action = data.get("action", "")
                if action == "advance":
                    session.roadmap.advance_phase()
                elif action == "complete":
                    session.roadmap.complete_item(data.get("item_id", ""))
                await ws.send_json(
                    {
                        "type": "roadmap_update",
                        "data": _serialize(session.roadmap.state),
                    }
                )

            elif msg_type == "ping":
                await ws.send_json({"type": "pong", "ts": time.time()})

    except WebSocketDisconnect:
        pass


async def _check_coaching_triggers(session, detection, ws):
    """Delegate to ws_session module."""
    await check_coaching_triggers(session, detection, ws, knowledge_base, _serialize)


def _treatment_text_to_trigger(text):
    """Delegate to ws_session module."""
    return treatment_text_to_trigger(text)


# ─── Collaboration Endpoints ─────────────────────────────────────────────────


class CreateRoomRequest(BaseModel):
    username: str = "guest"

@app.post("/api/collab/create")
async def create_collab_room(req: CreateRoomRequest = None) -> dict:
    """Create a new collaboration room based on billing tier."""
    username = req.username if req else "guest"
    tier = billing_manager.verify_account_tier(username)

    # We allow standard users to create basic rooms, but flag premium context
    room_id = await collab_manager.create_room()
    return {
        "room_id": room_id,
        "ws_url": f"/ws/collab/{room_id}",
        "tier_granted": tier
    }

class StripeWebhookPayload(BaseModel):
    type: str
    data: dict

@app.post("/api/billing/webhook")
async def stripe_webhook(payload: StripeWebhookPayload) -> dict:
    success = billing_manager.handle_webhook(payload.dict())
    return {"status": "processed", "success": success}


@app.get("/api/collab/rooms")
async def list_collab_rooms() -> dict:
    return {
        "active_rooms": collab_manager.room_count(),
        "active_peers": collab_manager.active_peers(),
    }


@app.websocket("/ws/collab/{room_id}")
async def collab_ws(ws: WebSocket, room_id: str) -> None:
    """Collaboration WebSocket handled by collab_router."""
    token = ws.query_params.get("token")
    user_payload = auth.verify_token(token) if token else None

    if not user_payload:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await handle_collab_ws(
        ws,
        room_id,
        collab_manager,
        webrtc_manager,
        global_ddsp_decoder,
        user_payload
    )
