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

import os

from dotenv import load_dotenv

if os.getenv("CLOUD_NATIVE_WORKSPACE", "false").lower() != "true":
    load_dotenv()

from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from claudio.collab.session_manager import SessionManager
from claudio.collab.webrtc_manager import WebRTCManager
from claudio.intent.intent_decoder import IntentDecoder
from claudio.server import auth
from claudio.server.billing import billing_manager
from claudio.server.collab_router import handle_collab_ws

app = FastAPI(title="Claudio Intelligence Server", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Shared Engine Instances ──────────────────────────────────────────────────

collab_manager = SessionManager()
webrtc_manager = WebRTCManager(collab_manager)

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
        "version": "3.0.0",
        "architecture": "pure-intent",
    }


# ─── Audio WebSocket: real neural codec (EnCodec) ────────────────────────────


# ─── Removed Bloat Endpoints ─────────────────────────────────────────────────

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
    return {"room_id": room_id, "ws_url": f"/ws/collab/{room_id}", "tier_granted": tier}


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

    await handle_collab_ws(ws, room_id, collab_manager, webrtc_manager, global_ddsp_decoder, user_payload)
