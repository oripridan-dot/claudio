"""
test_collab_e2e.py — End-to-End Collaboration & Network Tests

Tests the full capture→serialize→network→deserialize→regenerate pipeline
and collaboration infrastructure under realistic conditions:
  - Full wire roundtrip fidelity
  - Frontend 34-byte packet compatibility
  - Delta compression MFCC carry-forward
  - Multi-peer concurrent broadcast
  - Stale room cleanup
  - Room capacity limits
  - Network simulation: packet loss, reordering
"""
from __future__ import annotations

import struct
import time

import numpy as np
import pytest

from claudio.collab.session_manager import (
    SessionManager,
)
from claudio.intent.intent_decoder import IntentDecoder
from claudio.intent.intent_encoder import IntentEncoder
from claudio.intent.intent_protocol import (
    IntentPacket,
    IntentStream,
    PacketFlags,
)

SAMPLE_RATE = 44_100


def _sine(freq: float, dur: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float64) / SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


class MockWebSocket:
    """Minimal WebSocket mock for testing broadcast logic."""

    def __init__(self, *, fail_after: int = -1) -> None:
        self.sent_bytes: list[bytes] = []
        self.sent_json: list[dict] = []
        self.closed = False
        self._fail_after = fail_after  # Raise after N sends
        self._send_count = 0

    async def send_bytes(self, data: bytes) -> None:
        self._send_count += 1
        if 0 < self._fail_after <= self._send_count:
            raise ConnectionError("Simulated disconnect")
        self.sent_bytes.append(data)

    async def send_json(self, data: dict) -> None:
        self._send_count += 1
        if 0 < self._fail_after <= self._send_count:
            raise ConnectionError("Simulated disconnect")
        self.sent_json.append(data)

    async def close(self) -> None:
        self.closed = True


# ═══════════════════════════════════════════════════════════════════════
# Test: Full Wire Roundtrip (Encode → Serialize → Deserialize → Decode)
# ═══════════════════════════════════════════════════════════════════════

class TestWireRoundtrip:
    """Verify full pipeline fidelity through the wire format."""

    def test_full_roundtrip_preserves_f0(self) -> None:
        """F0 should survive encode→pack→bytes→unpack→decode with zero error."""
        enc = IntentEncoder(sample_rate=SAMPLE_RATE)
        stream = IntentStream()

        audio = _sine(440, 0.5)
        frames = enc.encode_block(audio)

        wire_frames = []
        for f in frames:
            pkt = stream.pack(f)
            data = pkt.to_bytes()
            restored = IntentPacket.from_bytes(data)
            if restored.frame:
                wire_frames.append(restored.frame)

        # F0 should be lossless (float32 roundtrip)
        voiced_orig = [f for f in frames if f.f0_hz > 0]
        voiced_wire = [f for f in wire_frames if f.f0_hz > 0]
        assert len(voiced_wire) > 0, "No voiced frames after wire roundtrip"

        for orig, wire in zip(voiced_orig, voiced_wire, strict=False):
            assert abs(orig.f0_hz - wire.f0_hz) < 0.01, (
                f"F0 drift: {orig.f0_hz} → {wire.f0_hz}"
            )

    def test_delta_frames_carry_forward_mfcc(self) -> None:
        """Decoder should use carried-forward MFCCs for delta frames."""
        enc = IntentEncoder(sample_rate=SAMPLE_RATE)
        stream = IntentStream()
        dec = IntentDecoder(sample_rate=SAMPLE_RATE)

        audio = _sine(440, 1.0)
        frames = enc.encode_block(audio)

        wire_frames = []
        for f in frames:
            pkt = stream.pack(f)
            restored = IntentPacket.from_bytes(pkt.to_bytes())
            if restored.frame:
                wire_frames.append(restored.frame)

        # Count delta frames (empty MFCCs) — verify delta is active
        assert any(not f.mfcc or len(f.mfcc) < 2 for f in wire_frames)

        # Decode through wire frames — should use carry-forward MFCCs
        decoded = dec.decode_frames(wire_frames)

        # Verify decoder has remembered the MFCCs
        assert len(dec._last_mfcc) >= 2, (
            "Decoder did not carry forward MFCCs from full frame"
        )
        assert len(decoded) > 0, "Decoded audio is empty"

    def test_roundtrip_audio_quality(self) -> None:
        """Roundtrip audio should have detectable pitch content."""
        enc = IntentEncoder(sample_rate=SAMPLE_RATE)
        stream = IntentStream()
        dec = IntentDecoder(sample_rate=SAMPLE_RATE)

        audio = _sine(440, 0.5, amp=0.3)
        frames = enc.encode_block(audio)

        wire_frames = []
        for f in frames:
            pkt = stream.pack(f)
            restored = IntentPacket.from_bytes(pkt.to_bytes())
            if restored.frame:
                wire_frames.append(restored.frame)

        decoded = dec.decode_frames(wire_frames)
        rms = float(np.sqrt(np.mean(decoded ** 2)))

        # Should not be silent
        assert rms > 0.01, f"Roundtrip audio too quiet: rms={rms:.4f}"


# ═══════════════════════════════════════════════════════════════════════
# Test: Frontend 34-byte Packet Compatibility
# ═══════════════════════════════════════════════════════════════════════

class TestFrontendCompatibility:
    """Verify backend can parse frontend's 34-byte packets."""

    def _make_frontend_packet(
        self, seq: int, ts: float, f0: float, conf: float,
        db: float, norm: float, centroid: float,
        onset: bool = False, onset_str: float = 0.0,
    ) -> bytes:
        """Simulate a frontend-generated 34-byte packet."""
        buf = bytearray(34)
        flags = 0x08 if norm < 0.01 else (0x05 if onset else 0x01)
        struct.pack_into("<IfB", buf, 0, seq & 0xFFFFFFFF, ts, flags)
        struct.pack_into(
            "<fffffBf", buf, 9,
            f0, conf, db, norm, centroid,
            1 if onset else 0, onset_str,
        )
        return bytes(buf)

    def test_parse_voiced_packet(self) -> None:
        """Backend should parse a voiced 34-byte packet correctly."""
        pkt = IntentPacket.from_bytes(
            self._make_frontend_packet(1, 100.0, 440.0, 0.95, -12.0, 0.85, 2500.0)
        )
        assert pkt.frame is not None
        assert abs(pkt.frame.f0_hz - 440.0) < 0.1
        assert abs(pkt.frame.loudness_db - (-12.0)) < 0.1
        assert pkt.frame.mfcc == []  # No MFCCs in 34-byte packet

    def test_parse_silence_packet(self) -> None:
        """Backend should parse a silence 34-byte packet correctly."""
        pkt = IntentPacket.from_bytes(
            self._make_frontend_packet(2, 200.0, 0, 0, -80.0, 0.0, 0)
        )
        assert pkt.frame is None  # Silence packet has no frame
        assert pkt.flags & PacketFlags.SILENCE

    def test_parse_onset_packet(self) -> None:
        """Backend should parse an onset 34-byte packet correctly."""
        pkt = IntentPacket.from_bytes(
            self._make_frontend_packet(3, 300.0, 440.0, 0.9, -10.0, 0.9, 3000.0,
                                       onset=True, onset_str=0.6)
        )
        assert pkt.frame is not None
        assert pkt.frame.is_onset is True
        assert pkt.frame.onset_strength > 0.5

    def test_decode_frontend_packet(self) -> None:
        """Decoder should handle a frontend packet (no MFCCs) gracefully."""
        pkt = IntentPacket.from_bytes(
            self._make_frontend_packet(1, 100.0, 440.0, 0.95, -12.0, 0.85, 2500.0)
        )
        dec = IntentDecoder(sample_rate=SAMPLE_RATE)
        audio = dec.decode_frames([pkt.frame])
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))


# ═══════════════════════════════════════════════════════════════════════
# Test: Network Simulation (packet loss, reordering)
# ═══════════════════════════════════════════════════════════════════════

class TestNetworkSimulation:
    """Verify pipeline handles network impairments gracefully."""

    def test_packet_loss_resilience(self) -> None:
        """Pipeline should handle 10% packet loss without crashing."""
        enc = IntentEncoder(sample_rate=SAMPLE_RATE)
        stream = IntentStream()
        dec = IntentDecoder(sample_rate=SAMPLE_RATE)

        audio = _sine(440, 1.0)
        frames = enc.encode_block(audio)

        # Serialize all, then drop 10%
        rng = np.random.default_rng(42)
        wire_frames = []
        dropped = 0
        for f in frames:
            pkt = stream.pack(f)
            data = pkt.to_bytes()
            if rng.random() < 0.1:
                dropped += 1
                continue  # Drop packet
            restored = IntentPacket.from_bytes(data)
            if restored.frame:
                wire_frames.append(restored.frame)

        decoded = dec.decode_frames(wire_frames)
        assert len(decoded) > 0, "All audio lost to packet loss"
        assert dropped > 0, "No packets were dropped (test is invalid)"

    def test_out_of_order_packets(self) -> None:
        """Pipeline should handle minor reordering without crashing."""
        enc = IntentEncoder(sample_rate=SAMPLE_RATE)
        stream = IntentStream()
        dec = IntentDecoder(sample_rate=SAMPLE_RATE)

        audio = _sine(440, 0.5)
        frames = enc.encode_block(audio)

        wire_frames = []
        for f in frames:
            pkt = stream.pack(f)
            restored = IntentPacket.from_bytes(pkt.to_bytes())
            if restored.frame:
                wire_frames.append(restored.frame)

        # Swap adjacent pairs
        for i in range(0, len(wire_frames) - 1, 4):
            wire_frames[i], wire_frames[i + 1] = wire_frames[i + 1], wire_frames[i]

        decoded = dec.decode_frames(wire_frames)
        assert len(decoded) > 0


# ═══════════════════════════════════════════════════════════════════════
# Test: Multi-Peer Concurrent Operations
# ═══════════════════════════════════════════════════════════════════════

class TestMultiPeerOperations:
    """Verify session manager handles concurrent peer operations."""

    @pytest.mark.asyncio
    async def test_broadcast_during_disconnect(self) -> None:
        """Broadcast should survive a peer disconnecting mid-iteration."""
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws1 = MockWebSocket()
        ws2 = MockWebSocket(fail_after=1)  # Fails on first send
        ws3 = MockWebSocket()

        p1 = await mgr.join_room(room_id, ws1, "Alice")
        await mgr.join_room(room_id, ws2, "Bob")  # Will fail during broadcast
        await mgr.join_room(room_id, ws3, "Carol")

        # Alice broadcasts — Bob's send fails, Carol should still receive
        await mgr.broadcast_intent(
            room_id, p1.peer_id, b"\x01\x02\x03",
        )
        # Bob's send raised but was caught
        assert len(ws3.sent_bytes) == 1, "Carol didn't receive packet"

    @pytest.mark.asyncio
    async def test_rapid_join_leave_cycle(self) -> None:
        """Rapid join/leave cycles should not corrupt room state."""
        mgr = SessionManager()

        for i in range(20):
            # Room gets auto-deleted when empty, so re-create each cycle
            room_id = await mgr.create_room()
            ws = MockWebSocket()
            peer = await mgr.join_room(room_id, ws, f"Peer{i}")
            assert peer is not None
            await mgr.leave_room(room_id, peer.peer_id)

        # All rooms should be cleaned up (empty → deleted)
        assert mgr.room_count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_stale_rooms(self) -> None:
        """Stale empty rooms should be cleaned up by TTL."""
        mgr = SessionManager()
        room_id = await mgr.create_room()

        # Manually backdate the room's creation
        room = mgr.get_room(room_id)
        room.created_at = time.time() - 7200  # 2 hours ago

        removed = await mgr.cleanup_stale_rooms(max_age_seconds=3600)
        assert removed == 1
        assert mgr.room_count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_preserves_active_rooms(self) -> None:
        """Stale cleanup should NOT remove rooms with active peers."""
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws = MockWebSocket()
        await mgr.join_room(room_id, ws, "Alice")

        # Backdate
        room = mgr.get_room(room_id)
        room.created_at = time.time() - 7200

        removed = await mgr.cleanup_stale_rooms(max_age_seconds=3600)
        assert removed == 0, "Should not remove room with active peer"
        assert mgr.room_count() == 1

    @pytest.mark.asyncio
    async def test_max_rooms_eviction(self) -> None:
        """Creating rooms beyond MAX should evict oldest empty rooms."""
        mgr = SessionManager()
        room_ids = []
        for _ in range(50):
            rid = await mgr.create_room()
            room_ids.append(rid)

        assert mgr.room_count() == 50

        # 51st room should trigger eviction of oldest empty
        rid = await mgr.create_room()
        assert mgr.room_count() <= 50

    @pytest.mark.asyncio
    async def test_broadcast_json_with_snapshot(self) -> None:
        """broadcast_json should work even if a peer disconnects."""
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws1 = MockWebSocket()
        ws2 = MockWebSocket(fail_after=1)  # Fails on first JSON send
        ws3 = MockWebSocket()

        await mgr.join_room(room_id, ws1, "Alice")
        await mgr.join_room(room_id, ws2, "Bob")
        await mgr.join_room(room_id, ws3, "Carol")

        # Should not raise despite Bob's failure
        await mgr.broadcast_json(room_id, None, {"type": "test"})
        assert len(ws1.sent_json) == 1
        assert len(ws3.sent_json) == 1


# ═══════════════════════════════════════════════════════════════════════
# Test: Protocol rms_energy Serialization
# ═══════════════════════════════════════════════════════════════════════

class TestRmsEnergySerialization:
    """Verify rms_energy survives the wire format."""

    def test_rms_energy_roundtrip(self) -> None:
        """rms_energy should survive to_bytes/from_bytes."""
        from claudio.intent.intent_encoder import IntentFrame

        frame = IntentFrame(
            timestamp_ms=0, f0_hz=440, f0_confidence=0.9,
            loudness_db=-12, loudness_norm=0.85,
            spectral_centroid_hz=2500, mfcc=[1.0] * 13,
            rms_energy=0.42,
        )
        pkt = IntentPacket(1, 0, PacketFlags.FULL_FRAME, frame)
        data = pkt.to_bytes()
        restored = IntentPacket.from_bytes(data)

        assert restored.frame is not None
        assert abs(restored.frame.rms_energy - 0.42) < 0.001, (
            f"rms_energy lost: {restored.frame.rms_energy}"
        )

    def test_rms_energy_backward_compat(self) -> None:
        """Old 94-byte packets (no rms_energy) should decode with rms=0."""
        # Build a 94-byte packet manually (no rms_energy field)
        header = struct.pack("<IfB", 1, 100.0, int(PacketFlags.FULL_FRAME))
        frame_data = struct.pack(
            "<fffffBf",
            440.0, 0.9, -12.0, 0.85, 2500.0, 0, 0.0,
        )
        mfcc_data = np.ones(13, dtype=np.float32).tobytes()
        vib_data = struct.pack("<ff", 0.0, 0.0)
        # 9 + 25 + 52 + 8 = 94 bytes — no rms_energy
        data_94 = header + frame_data + mfcc_data + vib_data

        pkt = IntentPacket.from_bytes(data_94)
        assert pkt.frame is not None
        assert pkt.frame.rms_energy == 0.0  # Default when not present


# ═══════════════════════════════════════════════════════════════════════
# Test: Rate Limiting
# ═══════════════════════════════════════════════════════════════════════

class TestRateLimiting:
    """Verify per-sender token bucket rate limiting."""

    @pytest.mark.asyncio
    async def test_burst_traffic_gets_throttled(self) -> None:
        """Sending huge packets rapidly should trigger rate limiting."""
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws_sender = MockWebSocket()
        ws_receiver = MockWebSocket()

        sender = await mgr.join_room(room_id, ws_sender, "Spammer")
        await mgr.join_room(room_id, ws_receiver, "Listener")

        # Send a burst of large packets far exceeding 64KB/s
        big_packet = b"\x00" * 8192  # 8KB per packet
        dropped = 0
        for _ in range(100):  # 100 × 8KB = 800KB in one go
            result = await mgr.broadcast_intent(
                room_id, sender.peer_id, big_packet,
            )
            if result == 0:
                dropped += 1

        assert dropped > 0, "Rate limiter never triggered on 800KB burst"

    @pytest.mark.asyncio
    async def test_normal_traffic_passes(self) -> None:
        """Normal intent packet rate (250Hz × 98B) should pass through."""
        mgr = SessionManager()
        room_id = await mgr.create_room()

        ws_sender = MockWebSocket()
        ws_receiver = MockWebSocket()

        sender = await mgr.join_room(room_id, ws_sender, "Alice")
        await mgr.join_room(room_id, ws_receiver, "Bob")

        # Simulate a single normal packet
        normal_packet = b"\x00" * 98
        result = await mgr.broadcast_intent(
            room_id, sender.peer_id, normal_packet,
        )
        assert result == 1, "Normal packet was incorrectly rate-limited"

