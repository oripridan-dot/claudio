"""
test_collab_session.py — Collaboration Room and Session Management Tests

Tests the collaboration infrastructure:
  - Room creation and lifecycle
  - Peer join/leave management
  - Intent packet broadcasting
  - Metrics tracking
  - Server endpoint integration
"""

from __future__ import annotations

import pytest

from claudio.collab.session_manager import (
    PeerRole,
    SessionManager,
)

# ═══════════════════════════════════════════════════════════════════════
# Mock WebSocket for testing
# ═══════════════════════════════════════════════════════════════════════


class MockWebSocket:
    """Minimal WebSocket mock for testing broadcast logic."""

    def __init__(self) -> None:
        self.sent_bytes: list[bytes] = []
        self.sent_json: list[dict] = []
        self.closed = False

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def send_json(self, data: dict) -> None:
        self.sent_json.append(data)

    async def close(self) -> None:
        self.closed = True


# ═══════════════════════════════════════════════════════════════════════
# Test: Session Manager
# ═══════════════════════════════════════════════════════════════════════


class TestSessionManager:
    """Test room lifecycle and peer management."""

    @pytest.mark.asyncio
    async def test_create_room(self) -> None:
        mgr = SessionManager()
        room_id = await mgr.create_room()
        assert len(room_id) == 8
        assert mgr.room_count() == 1

    @pytest.mark.asyncio
    async def test_join_room(self) -> None:
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws = MockWebSocket()
        peer = await mgr.join_room(room_id, ws, "Alice")
        assert peer is not None
        assert peer.display_name == "Alice"
        assert mgr.active_peers() == 1

    @pytest.mark.asyncio
    async def test_join_nonexistent_room(self) -> None:
        mgr = SessionManager()
        ws = MockWebSocket()
        peer = await mgr.join_room("invalid", ws)
        assert peer is None

    @pytest.mark.asyncio
    async def test_leave_room_cleanup(self) -> None:
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws = MockWebSocket()
        peer = await mgr.join_room(room_id, ws, "Alice")
        assert peer is not None

        await mgr.leave_room(room_id, peer.peer_id)
        assert mgr.active_peers() == 0
        # Empty room is cleaned up
        assert mgr.room_count() == 0

    @pytest.mark.asyncio
    async def test_max_peers(self) -> None:
        mgr = SessionManager()
        room_id = await mgr.create_room()

        # Fill the room to MAX_PEERS
        for i in range(8):
            ws = MockWebSocket()
            peer = await mgr.join_room(room_id, ws, f"Peer{i}")
            assert peer is not None

        # 9th peer should be rejected
        ws = MockWebSocket()
        peer = await mgr.join_room(room_id, ws, "Overflow")
        assert peer is None


# ═══════════════════════════════════════════════════════════════════════
# Test: Intent Broadcasting
# ═══════════════════════════════════════════════════════════════════════


class TestIntentBroadcast:
    """Test binary intent packet broadcasting between peers."""

    @pytest.mark.asyncio
    async def test_broadcast_to_other_peers(self) -> None:
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()

        p1 = await mgr.join_room(room_id, ws1, "Alice")
        await mgr.join_room(room_id, ws2, "Bob")
        await mgr.join_room(room_id, ws3, "Carol")

        # Alice sends — Bob and Carol should receive
        test_data = b"\x01\x02\x03\x04"
        sent = await mgr.broadcast_intent(room_id, p1.peer_id, test_data)

        assert sent == 2
        assert test_data in ws2.sent_bytes
        assert test_data in ws3.sent_bytes
        assert test_data not in ws1.sent_bytes  # Sender doesn't get own packets

    @pytest.mark.asyncio
    async def test_broadcast_updates_metrics(self) -> None:
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        p1 = await mgr.join_room(room_id, ws1, "Alice")
        p2 = await mgr.join_room(room_id, ws2, "Bob")

        # Send 10 packets
        for _ in range(10):
            await mgr.broadcast_intent(room_id, p1.peer_id, b"\x00" * 100)

        room = mgr.get_room(room_id)
        assert room is not None
        assert p1.packets_sent == 10
        assert p2.packets_received == 10

        metrics = room.metrics()
        assert metrics.total_packets == 10
        assert metrics.bytes_transmitted == 1000
        assert metrics.peer_count == 2


# ═══════════════════════════════════════════════════════════════════════
# Test: Room Metrics
# ═══════════════════════════════════════════════════════════════════════


class TestRoomMetrics:
    """Test room metrics tracking."""

    @pytest.mark.asyncio
    async def test_peer_list_format(self) -> None:
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws1 = MockWebSocket()
        await mgr.join_room(room_id, ws1, "Alice", PeerRole.SENDER)

        room = mgr.get_room(room_id)
        peer_list = room.peer_list()

        assert len(peer_list) == 1
        assert peer_list[0]["display_name"] == "Alice"
        assert peer_list[0]["role"] == "sender"

    @pytest.mark.asyncio
    async def test_broadcast_json(self) -> None:
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        await mgr.join_room(room_id, ws1, "Alice")
        await mgr.join_room(room_id, ws2, "Bob")

        # Broadcast JSON to all
        await mgr.broadcast_json(room_id, None, {"type": "test", "msg": "hello"})

        assert len(ws1.sent_json) == 1
        assert len(ws2.sent_json) == 1
        assert ws1.sent_json[0]["type"] == "test"
