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
import json
import time
import os
from dotenv import load_dotenv

if os.getenv("CLOUD_NATIVE_WORKSPACE", "false").lower() != "true":
    load_dotenv()

from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from claudio.collab.session_manager import PeerRole, SessionManager
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
from claudio.intent.intent_protocol import IntentPacket
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
from claudio.server.ws_session import (
    SessionState,
    check_coaching_triggers,
    treatment_text_to_trigger,
)
from claudio.server.billing import billing_manager

app = FastAPI(title="Claudio Intelligence Server", version="1.1.0")

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

# Global DDSP Decoder (Loaded if checkpoint exists)
try:
    global_ddsp_decoder = IntentDecoder(sample_rate=44100, model_path="checkpoints/forge_model_best.pt")
    print("[DDSP] Neural Decoder loaded on backend.")
except Exception as e:
    global_ddsp_decoder = None
    print(f"[DDSP] Failed to load Neural Decoder: {e}")


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
        "version": "1.1.0",
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
    """Collaboration WebSocket: binary intent packets + JSON signaling.

    Protocol:
      - Binary frames: raw intent packets, broadcast to all other peers
      - JSON frames:   signaling (join, leave, ping, instrument_set, metrics)
    """
    token = ws.query_params.get("token")
    user_payload = auth.verify_token(token) if token else None

    if not user_payload:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws.accept()

    # Parse query params
    params = dict(ws.query_params)
    display_name = user_payload.get("sub", params.get("name", "Musician"))
    role = PeerRole(params.get("role", "both"))

    # Join room
    peer = await collab_manager.join_room(room_id, ws, display_name, role)
    if peer is None:
        await ws.send_json({"type": "error", "message": "Room full or not found"})
        await ws.close()
        return

    # Notify room of new peer
    room = collab_manager.get_room(room_id)
    await collab_manager.broadcast_json(
        room_id,
        None,
        {
            "type": "peer_joined",
            "peer_id": peer.peer_id,
            "display_name": peer.display_name,
            "peers": room.peer_list() if room else [],
        },
    )

    # Send welcome
    await ws.send_json(
        {
            "type": "welcome",
            "peer_id": peer.peer_id,
            "room_id": room_id,
            "peers": room.peer_list() if room else [],
        }
    )

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.receive":
                if "bytes" in message and message["bytes"]:
                    # Binary frame: intent packet → broadcast
                    await collab_manager.broadcast_intent(
                        room_id,
                        peer.peer_id,
                        message["bytes"],
                    )

                    # Store for DDSP batching if enabled
                    if peer.ddsp_enabled and global_ddsp_decoder:
                        packet = IntentPacket.from_bytes(message["bytes"])
                        if packet.frame is not None:
                            peer._ddsp_buffer.append(packet.frame)

                        # Default batching: 12 frames at 120Hz = 100ms audio chunk
                        if len(peer._ddsp_buffer) >= 12:
                            # Run synthesis
                            audio_chunk = global_ddsp_decoder.decode_frames(peer._ddsp_buffer)
                            peer._ddsp_buffer.clear()

                            # Encode as binary: header("DDSP") + peer_id(8) + Float32Array
                            pid_b = peer.peer_id.encode("utf-8").ljust(8, b"\x00")[:8]
                            audio_bytes = audio_chunk.tobytes()
                            out_data = b"DDSP" + pid_b + audio_bytes

                            # Broadcast audio
                            room = collab_manager.get_room(room_id)
                            if room:
                                for opid, opeer in list(room.peers.items()):
                                    if opid != peer.peer_id:
                                        with contextlib.suppress(Exception):
                                            await opeer.ws.send_bytes(out_data)
                elif "text" in message and message["text"]:
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue  # Ignore malformed JSON
                    msg_type = data.get("type", "")

                    if msg_type == "ping":
                        await ws.send_json(
                            {
                                "type": "pong",
                                "ts": time.time(),
                                "server_ts": time.time(),
                            }
                        )

                    elif msg_type == "latency_report":
                        peer.latency_ms = data.get("latency_ms", 0.0)

                    elif msg_type == "instrument_set":
                        peer.instrument = data.get("instrument", "unknown")
                        room = collab_manager.get_room(room_id)
                        await collab_manager.broadcast_json(
                            room_id,
                            None,
                            {
                                "type": "peer_updated",
                                "peer_id": peer.peer_id,
                                "instrument": peer.instrument,
                                "peers": room.peer_list() if room else [],
                            },
                        )

                    elif msg_type == "ddsp_toggle":
                        peer.ddsp_enabled = data.get("enabled", False)
                        peer._ddsp_buffer = []  # reset buffer
                        if peer.ddsp_enabled and global_ddsp_decoder:
                            peer.instrument = "Neural DDSP"
                        else:
                            peer.instrument = "Additive JS"

                        room = collab_manager.get_room(room_id)
                        await collab_manager.broadcast_json(
                            room_id,
                            None,
                            {
                                "type": "peer_updated",
                                "peer_id": peer.peer_id,
                                "instrument": peer.instrument,
                                "peers": room.peer_list() if room else [],
                            },
                        )

                    elif msg_type == "metrics_request":
                        room = collab_manager.get_room(room_id)
                        if room:
                            m = room.metrics()
                            await ws.send_json(
                                {
                                    "type": "metrics",
                                    "peer_count": m.peer_count,
                                    "total_packets": m.total_packets,
                                    "bytes_transmitted": m.bytes_transmitted,
                                    "avg_latency_ms": round(m.avg_latency_ms, 1),
                                    "uptime_seconds": round(m.uptime_seconds, 1),
                                    "bandwidth_kbps": round(m.bandwidth_kbps, 2),
                                }
                            )

            elif message["type"] == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    finally:
        await collab_manager.leave_room(room_id, peer.peer_id)
        room = collab_manager.get_room(room_id)
        if room:
            await collab_manager.broadcast_json(
                room_id,
                None,
                {
                    "type": "peer_left",
                    "peer_id": peer.peer_id,
                    "peers": room.peer_list(),
                },
            )
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()
