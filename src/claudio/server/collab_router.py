"""
collab_router.py — Collaboration WebSocket Handler

Handles the multi-user collaboration loop, including intent broadcasting,
signaling for WebRTC, and session management.
extracted from claudio_server.py to satisfy 500-line limit.
"""

import contextlib
import json
import logging
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from claudio.collab.session_manager import PeerRole, SessionManager
from claudio.intent.intent_protocol import IntentPacket

logger = logging.getLogger(__name__)

async def handle_collab_ws(
    ws: WebSocket,
    room_id: str,
    collab_manager: SessionManager,
    webrtc_manager: Any,
    global_ddsp_decoder: Any,
    auth_payload: dict
) -> None:
    """Collaboration WebSocket: binary intent packets + JSON signaling."""
    await ws.accept()
    # Parse query params
    params = dict(ws.query_params)
    display_name = auth_payload.get("sub", params.get("name", "Musician"))
    role = PeerRole(params.get("role", "both"))
    instrument_model_url = params.get("instrument", "/models/ddsp_model.onnx")
    environment_ir = params.get("environment", "Studio_A")

    # Join room
    peer = await collab_manager.join_room(room_id, ws, display_name, role)
    if peer is None:
        await ws.send_json({"type": "error", "message": "Room full or not found"})
        await ws.close()
        return
        
    peer.instrument = instrument_model_url # Currently reusing the instrument field to map model URLs 
    peer.environment = environment_ir

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
                    # Build 8-byte UDP-style header for sender attribution
                    pid_b = peer.peer_id.encode("utf-8").ljust(8, b"\x00")[:8]
                    augmented_bytes = pid_b + message["bytes"]
                    pid_b = peer.peer_id.encode("utf-8").ljust(8, b"\x00")[:8]
                    augmented_bytes = pid_b + message["bytes"]

                    # 1. Broadcast via WebSocket (original path)
                    await collab_manager.broadcast_intent(
                        room_id,
                        peer.peer_id,
                        augmented_bytes,
                    )

                    # 2. Broadcast via WebRTC Data Channels (Integrated)
                    if hasattr(webrtc_manager, "broadcast_intent_p2p"):
                        await webrtc_manager.broadcast_intent_p2p(
                            room_id,
                            peer.peer_id,
                            augmented_bytes
                        )

                    # 3. Handle DDSP Neural Decoding state
                    if peer.ddsp_enabled and global_ddsp_decoder:
                        packet = IntentPacket.from_bytes(message["bytes"])
                        if packet.frame is not None:
                            peer._ddsp_buffer.append(packet.frame)

                        if len(peer._ddsp_buffer) >= 12:
                            audio_chunk = global_ddsp_decoder.decode_frames(peer._ddsp_buffer)
                            peer._ddsp_buffer.clear()

                            pid_b = peer.peer_id.encode("utf-8").ljust(8, b"\x00")[:8]
                            audio_bytes = audio_chunk.tobytes()
                            out_data = b"DDSP" + pid_b + audio_bytes

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
                        continue
                    msg_type = data.get("type", "")

                    if msg_type == "ping":
                        await ws.send_json({
                            "type": "pong",
                            "ts": time.time(),
                            "server_ts": time.time(),
                        })

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
                        peer._ddsp_buffer = []
                        peer.instrument = "Neural DDSP" if (peer.ddsp_enabled and global_ddsp_decoder) else "Additive JS"

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

                    elif msg_type == "webrtc_offer":
                        # Relay to all other peers in the room for p2p signaling
                        room = collab_manager.get_room(room_id)
                        if room:
                            for opid, opeer in list(room.peers.items()):
                                if opid != peer.peer_id:
                                    with contextlib.suppress(Exception):
                                        await opeer.ws.send_json({
                                            "type": "webrtc_offer",
                                            "from_peer": peer.peer_id,
                                            "sdp": data.get("sdp", ""),
                                            "rtc_type": data.get("rtc_type", "offer"),
                                        })

                    elif msg_type == "webrtc_answer":
                        # Relay answer back to the offering peer
                        target_id = data.get("to_peer", "")
                        room = collab_manager.get_room(room_id)
                        if room and target_id in room.peers:
                            with contextlib.suppress(Exception):
                                await room.peers[target_id].ws.send_json({
                                    "type": "webrtc_answer",
                                    "from_peer": peer.peer_id,
                                    "sdp": data.get("sdp", ""),
                                    "rtc_type": data.get("rtc_type", "answer"),
                                })

                    elif msg_type == "ice_candidate":
                        # Relay ICE candidate to all other peers in the room
                        room = collab_manager.get_room(room_id)
                        if room:
                            for opid, opeer in list(room.peers.items()):
                                if opid != peer.peer_id:
                                    with contextlib.suppress(Exception):
                                        await opeer.ws.send_json({
                                            "type": "ice_candidate",
                                            "from_peer": peer.peer_id,
                                            "candidate": data.get("candidate", ""),
                                            "sdpMid": data.get("sdpMid", ""),
                                            "sdpMLineIndex": data.get("sdpMLineIndex", 0),
                                        })

                    elif msg_type == "metrics_request":
                        room = collab_manager.get_room(room_id)
                        if room:
                            m = room.metrics()
                            await ws.send_json({
                                "type": "metrics",
                                "peer_count": m.peer_count,
                                "total_packets": m.total_packets,
                                "bytes_transmitted": m.bytes_transmitted,
                                "avg_latency_ms": round(m.avg_latency_ms, 1),
                                "uptime_seconds": round(m.uptime_seconds, 1),
                                "bandwidth_kbps": round(m.bandwidth_kbps, 2),
                            })

            elif message["type"] == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    finally:
        if hasattr(webrtc_manager, "close_peer"):
            await webrtc_manager.close_peer(peer.peer_id, room_id)
        else:
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
