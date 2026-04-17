"""
session_manager.py — Collaboration Room Management

Manages WebSocket-based collaboration rooms where peers stream
intent packets to each other in real-time.

Architecture:
  - Each room has a unique ID and up to MAX_PEERS participants
  - Peers send intent packets (binary) which are broadcast to all others
  - Room state (peer list, metrics) is broadcast as JSON
  - Signaling for WebRTC negotiation is handled via JSON messages

Room lifecycle:
  1. Creator opens room → gets room_id
  2. Joiner connects with room_id → added to peer list
  3. Both peers stream intent packets via binary WS frames
  4. On disconnect → peer removed, room cleaned if empty
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from fastapi import WebSocket


class PeerRole(StrEnum):
    SENDER = "sender"  # Streaming intent outward
    RECEIVER = "receiver"  # Receiving and regenerating
    BOTH = "both"  # Bidirectional (default for collab)


@dataclass
class PeerInfo:
    """Metadata about a connected peer."""

    peer_id: str
    display_name: str
    role: PeerRole
    ws: WebSocket
    instrument: str = "unknown"  # Reused as model_url alias for backwards compatibility
    environment: str = "Studio_A"
    joined_at: float = 0.0
    packets_sent: int = 0
    packets_received: int = 0
    last_packet_ts: float = 0.0
    latency_ms: float = 0.0  # Round-trip latency estimate
    # Rate limiting (token bucket): bytes allowed per second
    _rate_bucket: float = 0.0
    _rate_last_refill: float = 0.0
    # DDSP state
    ddsp_enabled: bool = False
    _ddsp_buffer: list = field(default_factory=list)


@dataclass
class RoomMetrics:
    """Live metrics for a collaboration room."""

    peer_count: int = 0
    total_packets: int = 0
    bytes_transmitted: int = 0
    avg_latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    bandwidth_kbps: float = 0.0


@dataclass
class CollabRoom:
    """A collaboration room where peers exchange intent packets."""

    room_id: str
    created_at: float = field(default_factory=time.time)
    peers: dict[str, PeerInfo] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _total_bytes: int = 0

    @property
    def peer_count(self) -> int:
        return len(self.peers)

    @property
    def is_empty(self) -> bool:
        return len(self.peers) == 0

    def metrics(self) -> RoomMetrics:
        now = time.time()
        latencies = [p.latency_ms for p in self.peers.values() if p.latency_ms > 0]
        uptime = now - self.created_at
        return RoomMetrics(
            peer_count=len(self.peers),
            total_packets=sum(p.packets_sent for p in self.peers.values()),
            bytes_transmitted=self._total_bytes,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            uptime_seconds=uptime,
            bandwidth_kbps=(self._total_bytes * 8 / 1024 / uptime) if uptime > 0 else 0.0,
        )

    def peer_list(self) -> list[dict]:
        return [
            {
                "peer_id": p.peer_id,
                "display_name": p.display_name,
                "role": p.role.value,
                "instrument": p.instrument,
                "model_url": p.instrument,
                "environment": getattr(p, "environment", "Studio_A"),
                "packets_sent": p.packets_sent,
                "latency_ms": round(p.latency_ms, 1),
            }
            for p in self.peers.values()
        ]


MAX_PEERS_PER_ROOM = 8
MAX_ROOMS = 50
# Rate limit: 64 KB/s per sender (generous for 250Hz × 98B = ~24KB/s)
RATE_LIMIT_BYTES_PER_SEC = 65_536
RATE_LIMIT_BURST = 4096  # Allow burst of 4KB


class SessionManager:
    """Manages all active collaboration rooms.

    Thread-safe: uses asyncio locks for concurrent WebSocket access.
    """

    def __init__(self) -> None:
        self._rooms: dict[str, CollabRoom] = {}
        self._lock = asyncio.Lock()

    async def create_room(self) -> str:
        """Create a new collaboration room. Returns room_id."""
        async with self._lock:
            if len(self._rooms) >= MAX_ROOMS:
                # Evict oldest empty room
                for rid, room in list(self._rooms.items()):
                    if room.is_empty:
                        del self._rooms[rid]
                        break

            room_id = uuid.uuid4().hex[:8]
            self._rooms[room_id] = CollabRoom(room_id=room_id)
            return room_id

    async def join_room(
        self,
        room_id: str,
        ws: WebSocket,
        display_name: str = "Musician",
        role: PeerRole = PeerRole.BOTH,
    ) -> PeerInfo | None:
        """Join a peer to a room. Returns PeerInfo or None if room full/missing."""
        async with self._lock:
            room = self._rooms.get(room_id)
            if room is None:
                return None
            if room.peer_count >= MAX_PEERS_PER_ROOM:
                return None

            peer_id = uuid.uuid4().hex[:6]
            peer = PeerInfo(
                peer_id=peer_id,
                display_name=display_name,
                role=role,
                ws=ws,
                joined_at=time.time(),
            )
            room.peers[peer_id] = peer
            return peer

    async def leave_room(self, room_id: str, peer_id: str) -> None:
        """Remove a peer from a room. Cleans up empty rooms."""
        async with self._lock:
            room = self._rooms.get(room_id)
            if room and peer_id in room.peers:
                del room.peers[peer_id]
                if room.is_empty:
                    del self._rooms[room_id]

    async def broadcast_intent(
        self,
        room_id: str,
        sender_id: str,
        data: bytes,
    ) -> int:
        """Broadcast binary intent packet to all peers except sender.

        Returns number of peers the packet was sent to.
        Rate-limited per sender via token bucket (64KB/s).
        Uses a snapshot of the peers dict to avoid RuntimeError
        if a peer disconnects (leave_room) during iteration.
        """
        room = self._rooms.get(room_id)
        if room is None:
            return 0

        sender = room.peers.get(sender_id)
        if sender:
            sender.packets_sent += 1
            sender.last_packet_ts = time.time()

            # Token bucket rate limiting
            now = time.time()
            elapsed = now - sender._rate_last_refill
            sender._rate_bucket = min(
                RATE_LIMIT_BURST,
                sender._rate_bucket + elapsed * RATE_LIMIT_BYTES_PER_SEC,
            )
            sender._rate_last_refill = now

            packet_cost = float(len(data))
            if sender._rate_bucket < packet_cost:
                return 0  # Rate limited — drop silently
            sender._rate_bucket -= packet_cost

        room._total_bytes += len(data)
        sent_count = 0

        # Snapshot to avoid dict-changed-during-iteration
        peers_snapshot = list(room.peers.items())
        for pid, peer in peers_snapshot:
            if pid == sender_id:
                continue
            try:
                await peer.ws.send_bytes(data)
                peer.packets_received += 1
                sent_count += 1
            except Exception:
                pass  # Peer disconnected — will be cleaned up

        return sent_count

    async def broadcast_json(
        self,
        room_id: str,
        sender_id: str | None,
        message: dict,
    ) -> None:
        """Broadcast a JSON message to all peers (including sender)."""
        room = self._rooms.get(room_id)
        if room is None:
            return

        # Snapshot to avoid dict-changed-during-iteration
        peers_snapshot = list(room.peers.values())
        for peer in peers_snapshot:
            with contextlib.suppress(Exception):
                await peer.ws.send_json(message)

    def get_room(self, room_id: str) -> CollabRoom | None:
        return self._rooms.get(room_id)

    def room_count(self) -> int:
        return len(self._rooms)

    def active_peers(self) -> int:
        return sum(r.peer_count for r in self._rooms.values())

    async def cleanup_stale_rooms(self, max_age_seconds: float = 3600) -> int:
        """Remove empty rooms older than max_age_seconds. Returns count removed."""
        now = time.time()
        removed = 0
        async with self._lock:
            stale = [
                rid for rid, room in self._rooms.items() if room.is_empty and (now - room.created_at) > max_age_seconds
            ]
            for rid in stale:
                del self._rooms[rid]
                removed += 1
        return removed
