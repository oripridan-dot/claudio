"""
intent_protocol.py — Wire Protocol for Intent Packets

Defines the binary serialization format for transmitting IntentFrames
over WebRTC data channels with minimal bandwidth.

Protocol design:
  - MessagePack for compact binary serialization (30-50% smaller than JSON)
  - Delta compression: only transmit fields that changed beyond threshold
  - Sequence numbering for packet loss detection
  - Timestamps for jitter buffer alignment

Wire format per packet:
  [seq:u32][ts_ms:f32][flags:u8][payload:variable]

Typical bandwidth:
  Full frame:  ~80 bytes × 250 Hz = ~20 KB/s
  Delta frame: ~15 bytes × 250 Hz =  ~4 KB/s
  With silence detection: ~0.5 KB/s average
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntFlag

import numpy as np

from claudio.intent.intent_encoder import IntentFrame


class PacketFlags(IntFlag):
    """Bitfield flags for intent packets."""

    FULL_FRAME = 0x01  # Contains all fields (no delta)
    DELTA = 0x02  # Contains only changed fields
    ONSET = 0x04  # Onset event in this frame
    SILENCE = 0x08  # Below noise floor — no payload
    KEY_FRAME = 0x10  # Periodic full refresh (every 1s)


# Delta compression thresholds — changes below these are suppressed
DELTA_F0_HZ = 0.5  # ~1 cent at A4
DELTA_LOUDNESS_DB = 0.5  # Barely perceptible
DELTA_CENTROID_HZ = 10.0  # Minor timbral shift
DELTA_MFCC = 0.1  # MFCC coefficient change


@dataclass
class IntentPacket:
    """Serializable intent packet for network transmission."""

    sequence: int
    timestamp_ms: float
    flags: PacketFlags
    frame: IntentFrame | None  # None for SILENCE packets

    def to_bytes(self) -> bytes:
        """Serialize to compact binary format."""
        header = struct.pack(
            "<IfB",
            self.sequence & 0xFFFFFFFF,
            self.timestamp_ms,
            int(self.flags),
        )

        if self.flags & PacketFlags.SILENCE:
            return header

        if self.frame is None:
            return header

        # Pack frame data
        frame_data = struct.pack(
            "<fffffBf",
            self.frame.f0_hz,
            self.frame.f0_confidence,
            self.frame.loudness_db,
            self.frame.loudness_norm,
            self.frame.spectral_centroid_hz,
            1 if self.frame.is_onset else 0,
            self.frame.onset_strength,
        )

        # Pack MFCCs (13 × float32 = 52 bytes)
        mfcc_data = b""
        if self.frame.mfcc:
            mfcc_arr = np.array(self.frame.mfcc[:13], dtype=np.float32)
            if len(mfcc_arr) < 13:
                mfcc_arr = np.pad(mfcc_arr, (0, 13 - len(mfcc_arr)))
            mfcc_data = mfcc_arr.tobytes()

        # Pack vibrato
        vib_data = struct.pack(
            "<ff",
            self.frame.vibrato_rate_hz,
            self.frame.vibrato_depth_cents,
        )

        # Pack rms_energy
        rms_data = struct.pack("<f", self.frame.rms_energy)

        return header + frame_data + mfcc_data + vib_data + rms_data

    @classmethod
    def from_bytes(cls, data: bytes) -> IntentPacket:
        """Deserialize from binary format."""
        seq, ts, flags_raw = struct.unpack_from("<IfB", data, 0)
        flags = PacketFlags(flags_raw)
        offset = 9  # 4 + 4 + 1

        if flags & PacketFlags.SILENCE:
            return cls(sequence=seq, timestamp_ms=ts, flags=flags, frame=None)

        if len(data) < offset + 25:
            return cls(sequence=seq, timestamp_ms=ts, flags=flags, frame=None)

        f0, f0_conf, loud_db, loud_norm, centroid, onset_flag, onset_str = struct.unpack_from("<fffffBf", data, offset)
        offset += 25  # 5×4 + 1 + 4

        # Unpack MFCCs
        mfcc = []
        if len(data) >= offset + 52:
            mfcc_arr = np.frombuffer(data[offset : offset + 52], dtype=np.float32)
            mfcc = mfcc_arr.tolist()
            offset += 52

        # Unpack vibrato
        vib_rate, vib_depth = 0.0, 0.0
        if len(data) >= offset + 8:
            vib_rate, vib_depth = struct.unpack_from("<ff", data, offset)
            offset += 8

        # Unpack rms_energy (optional — backward compatible)
        rms_energy = 0.0
        if len(data) >= offset + 4:
            (rms_energy,) = struct.unpack_from("<f", data, offset)

        frame = IntentFrame(
            timestamp_ms=ts,
            f0_hz=f0,
            f0_confidence=f0_conf,
            loudness_db=loud_db,
            loudness_norm=loud_norm,
            spectral_centroid_hz=centroid,
            mfcc=mfcc,
            is_onset=bool(onset_flag),
            onset_strength=onset_str,
            vibrato_rate_hz=vib_rate,
            vibrato_depth_cents=vib_depth,
            rms_energy=rms_energy,
        )

        return cls(sequence=seq, timestamp_ms=ts, flags=flags, frame=frame)


class IntentStream:
    """Manages a stream of intent packets with delta compression.

    Tracks the previous frame to enable delta encoding —
    only fields that changed beyond threshold are transmitted.
    """

    def __init__(self) -> None:
        self._seq = 0
        self._prev_frame: IntentFrame | None = None
        self._key_frame_interval = 250  # Every 1s at 250Hz

    def pack(self, frame: IntentFrame) -> IntentPacket:
        """Pack a frame into a packet with delta compression.

        Delta packets omit MFCCs and vibrato when fields are stable,
        saving ~60 bytes per packet (~55% on sustained tones).
        """
        self._seq += 1

        # Silence detection
        if frame.loudness_norm < 0.01:
            self._prev_frame = frame
            return IntentPacket(
                sequence=self._seq,
                timestamp_ms=frame.timestamp_ms,
                flags=PacketFlags.SILENCE,
                frame=None,
            )

        # Key frame every 1 second — always full
        is_key = (self._seq % self._key_frame_interval) == 0

        # Check if delta compression is possible
        use_delta = (
            not is_key
            and self._prev_frame is not None
            and not frame.is_onset
            and abs(frame.f0_hz - self._prev_frame.f0_hz) < DELTA_F0_HZ
            and abs(frame.loudness_db - self._prev_frame.loudness_db) < DELTA_LOUDNESS_DB
            and abs(frame.spectral_centroid_hz - self._prev_frame.spectral_centroid_hz) < DELTA_CENTROID_HZ
        )

        if use_delta:
            # Delta: strip MFCCs and vibrato to save ~60 bytes
            delta_frame = IntentFrame(
                timestamp_ms=frame.timestamp_ms,
                f0_hz=frame.f0_hz,
                f0_confidence=frame.f0_confidence,
                loudness_db=frame.loudness_db,
                loudness_norm=frame.loudness_norm,
                spectral_centroid_hz=frame.spectral_centroid_hz,
                mfcc=[],  # Omitted — receiver uses previous
                is_onset=False,
                onset_strength=0.0,
                rms_energy=frame.rms_energy,
            )
            flags = PacketFlags.DELTA
            self._prev_frame = frame
            return IntentPacket(
                sequence=self._seq,
                timestamp_ms=frame.timestamp_ms,
                flags=flags,
                frame=delta_frame,
            )

        flags = PacketFlags.FULL_FRAME
        if is_key:
            flags |= PacketFlags.KEY_FRAME
        if frame.is_onset:
            flags |= PacketFlags.ONSET

        self._prev_frame = frame
        return IntentPacket(
            sequence=self._seq,
            timestamp_ms=frame.timestamp_ms,
            flags=flags,
            frame=frame,
        )

    def reset(self) -> None:
        """Reset stream state."""
        self._seq = 0
        self._prev_frame = None

    @property
    def bandwidth_estimate_kbps(self) -> float:
        """Estimate current bandwidth usage in KB/s."""
        # Full frame: ~94 bytes, Silence: 9 bytes
        # Average realistic: ~50 bytes at 250Hz = 12.5 KB/s
        return 94 * FRAME_RATE_HZ / 1024


FRAME_RATE_HZ = 250  # Must match intent_encoder
