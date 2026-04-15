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

from dotenv import load_dotenv

if os.getenv("CLOUD_NATIVE_WORKSPACE", "false").lower() != "true":
    load_dotenv()

import asyncio
from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from claudio.codec.neural_codec import NeuralCodec
from claudio.collab.session_manager import SessionManager
from claudio.collab.webrtc_manager import WebRTCManager
from claudio.intent.intent_decoder import IntentDecoder
from claudio.server import auth
from claudio.server.billing import billing_manager
from claudio.server.collab_router import handle_collab_ws

app = FastAPI(title="Claudio Intelligence Server", version="1.2.0")

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
