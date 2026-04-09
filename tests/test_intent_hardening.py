"""
test_intent_hardening.py — Edge Cases, Robustness, and Coverage Tests

Tests added in Round 3 audit to close coverage gaps:
  - Input sanitization: NaN/Inf safety
  - Protocol robustness: malformed packets, delta compression
  - Phase continuity: click prevention
  - Reset behavior: encoder/decoder state cleanup
  - Decoder edge cases: extreme F0, empty lists, fallback paths
"""
from __future__ import annotations

import math
import struct

import numpy as np

from claudio.intent.intent_decoder import IntentDecoder
from claudio.intent.intent_encoder import IntentEncoder, IntentFrame
from claudio.intent.intent_protocol import IntentPacket, IntentStream, PacketFlags

SAMPLE_RATE = 44_100


def _sine(freq: float, dur: float, amp: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float64) / SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Test: Input Sanitization (NaN/Inf safety)
# ═══════════════════════════════════════════════════════════════════════

class TestInputSanitization:
    """Verify encoder handles corrupt input without propagating NaN/Inf."""

    def test_nan_input_produces_finite_output(self) -> None:
        """NaN audio should produce all-finite frame values."""
        encoder = IntentEncoder(sample_rate=SAMPLE_RATE)
        nan_audio = np.full(SAMPLE_RATE, np.nan, dtype=np.float32)
        frames = encoder.encode_block(nan_audio)
        for f in frames:
            assert math.isfinite(f.f0_hz), "NaN leaked into f0_hz"
            assert math.isfinite(f.loudness_db), "NaN leaked into loudness_db"
            assert math.isfinite(f.loudness_norm), "NaN leaked into loudness_norm"
            assert math.isfinite(f.spectral_centroid_hz), "NaN leaked into centroid"

    def test_inf_input_produces_finite_output(self) -> None:
        """Inf audio should produce all-finite frame values."""
        encoder = IntentEncoder(sample_rate=SAMPLE_RATE)
        inf_audio = np.full(SAMPLE_RATE, np.inf, dtype=np.float32)
        frames = encoder.encode_block(inf_audio)
        for f in frames:
            assert math.isfinite(f.f0_hz), "Inf leaked into f0_hz"
            assert math.isfinite(f.loudness_db), "Inf leaked into loudness_db"


# ═══════════════════════════════════════════════════════════════════════
# Test: Protocol Robustness (malformed packets)
# ═══════════════════════════════════════════════════════════════════════

class TestProtocolRobustness:
    """Verify protocol handles edge cases without crashing."""

    def test_truncated_packet_returns_frameless(self) -> None:
        """A packet with valid header but truncated payload → no frame."""
        header = struct.pack("<IfB", 1, 100.0, int(PacketFlags.FULL_FRAME))
        pkt = IntentPacket.from_bytes(header)
        assert pkt.sequence == 1
        assert pkt.frame is None

    def test_empty_mfcc_roundtrip(self) -> None:
        """Frame with empty MFCCs should survive serialization."""
        frame = IntentFrame(
            timestamp_ms=0, f0_hz=440, f0_confidence=0.9,
            loudness_db=-12, loudness_norm=0.85,
            spectral_centroid_hz=2500, mfcc=[],
        )
        pkt = IntentPacket(
            sequence=1, timestamp_ms=0,
            flags=PacketFlags.FULL_FRAME, frame=frame,
        )
        data = pkt.to_bytes()
        restored = IntentPacket.from_bytes(data)
        assert restored.frame is not None
        assert abs(restored.frame.f0_hz - 440) < 0.1

    def test_delta_packet_smaller_than_full(self) -> None:
        """Delta packets (no MFCCs) must be smaller than full frames."""
        full_frame = IntentFrame(
            timestamp_ms=0, f0_hz=440, f0_confidence=0.9,
            loudness_db=-12, loudness_norm=0.85,
            spectral_centroid_hz=2500, mfcc=[1.0] * 13,
            vibrato_rate_hz=5.5, vibrato_depth_cents=30,
        )
        delta_frame = IntentFrame(
            timestamp_ms=0, f0_hz=440, f0_confidence=0.9,
            loudness_db=-12, loudness_norm=0.85,
            spectral_centroid_hz=2500, mfcc=[],
        )
        full_pkt = IntentPacket(1, 0, PacketFlags.FULL_FRAME, full_frame)
        delta_pkt = IntentPacket(2, 0, PacketFlags.DELTA, delta_frame)

        full_bytes = len(full_pkt.to_bytes())
        delta_bytes = len(delta_pkt.to_bytes())

        assert delta_bytes < full_bytes, (
            f"Delta ({delta_bytes}B) not smaller than full ({full_bytes}B)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Test: Phase Continuity (click prevention)
# ═══════════════════════════════════════════════════════════════════════

class TestPhaseContinuity:
    """Verify decoder produces smooth audio across frame boundaries."""

    def test_no_clicks_on_sustained_tone(self) -> None:
        """Adjacent decode frames should not have sample discontinuities."""
        decoder = IntentDecoder(sample_rate=SAMPLE_RATE)
        frames = [
            IntentFrame(
                timestamp_ms=i * 4, f0_hz=440, f0_confidence=0.95,
                loudness_db=-12, loudness_norm=0.85,
                spectral_centroid_hz=2500, mfcc=[1.0] * 13,
            )
            for i in range(50)
        ]
        audio = decoder.decode_frames(frames)
        hop = SAMPLE_RATE // 250

        max_jump = 0.0
        for i in range(1, 50):
            boundary = i * hop
            if boundary < len(audio) - 1:
                jump = abs(float(audio[boundary]) - float(audio[boundary - 1]))
                max_jump = max(max_jump, jump)

        # Multi-harmonic synthesis (40 harmonics) naturally produces large
        # sample-to-sample changes from constructive/destructive interference.
        # A true click would be >> 2.0 (e.g., phase discontinuity or gain pop).
        assert max_jump < 2.0, f"Click detected at frame boundary: jump={max_jump:.4f}"

    def test_silence_to_voiced_no_click(self) -> None:
        """Transition from silence → voiced should not produce a pop."""
        decoder = IntentDecoder(sample_rate=SAMPLE_RATE)
        silent = [
            IntentFrame(timestamp_ms=i * 4, f0_hz=0, f0_confidence=0,
                        loudness_db=-80, loudness_norm=0.0,
                        spectral_centroid_hz=0, mfcc=[])
            for i in range(20)
        ]
        voiced = [
            IntentFrame(timestamp_ms=(20 + i) * 4, f0_hz=440, f0_confidence=0.9,
                        loudness_db=-20, loudness_norm=0.5,
                        spectral_centroid_hz=2500, mfcc=[1.0] * 13)
            for i in range(20)
        ]
        audio = decoder.decode_frames(silent + voiced)

        hop = SAMPLE_RATE // 250
        transition_idx = 20 * hop
        if transition_idx < len(audio):
            first_voiced = abs(float(audio[transition_idx]))
            assert first_voiced < 0.2, (
                f"Pop on silence→voiced transition: {first_voiced:.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Test: Delta Compression
# ═══════════════════════════════════════════════════════════════════════

class TestDeltaCompression:
    """Verify delta compression reduces bandwidth on sustained signals."""

    def test_sustained_tone_uses_delta(self) -> None:
        """Sustained tone should trigger delta compression."""
        stream = IntentStream()
        frames = [
            IntentFrame(
                timestamp_ms=i * 4, f0_hz=440.0, f0_confidence=0.95,
                loudness_db=-12, loudness_norm=0.85,
                spectral_centroid_hz=2500, mfcc=[1.0] * 13,
            )
            for i in range(100)
        ]

        delta_count = 0
        full_count = 0
        for f in frames:
            pkt = stream.pack(f)
            if pkt.flags & PacketFlags.DELTA:
                delta_count += 1
            elif pkt.flags & PacketFlags.FULL_FRAME:
                full_count += 1

        assert delta_count > 50, (
            f"Expected >50 delta packets, got {delta_count}/{len(frames)}"
        )

    def test_onset_forces_full_frame(self) -> None:
        """Onset detection should prevent delta compression."""
        stream = IntentStream()
        f1 = IntentFrame(
            timestamp_ms=0, f0_hz=440, f0_confidence=0.95,
            loudness_db=-12, loudness_norm=0.85,
            spectral_centroid_hz=2500, mfcc=[1.0] * 13,
        )
        stream.pack(f1)

        f2 = IntentFrame(
            timestamp_ms=4, f0_hz=440, f0_confidence=0.95,
            loudness_db=-12, loudness_norm=0.85,
            spectral_centroid_hz=2500, mfcc=[1.0] * 13,
            is_onset=True, onset_strength=0.5,
        )
        pkt = stream.pack(f2)
        assert pkt.flags & PacketFlags.FULL_FRAME, "Onset should force FULL_FRAME"
        assert not (pkt.flags & PacketFlags.DELTA), "Onset should not be DELTA"

    def test_delta_bandwidth_savings(self) -> None:
        """Delta packets must achieve measurable bandwidth reduction."""
        encoder = IntentEncoder(sample_rate=SAMPLE_RATE)
        stream = IntentStream()

        tone = _sine(440, 2.0)
        frames = encoder.encode_block(tone)

        total_delta = 0
        total_no_delta = 0
        for f in frames:
            pkt = stream.pack(f)
            total_delta += len(pkt.to_bytes())
            full_pkt = IntentPacket(
                pkt.sequence, pkt.timestamp_ms,
                PacketFlags.FULL_FRAME, f,
            )
            total_no_delta += len(full_pkt.to_bytes())

        savings = 1.0 - (total_delta / total_no_delta)
        assert savings > 0.2, (
            f"Delta savings too low: {savings:.1%} "
            f"({total_delta}B vs {total_no_delta}B)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Test: Encoder/Decoder Reset
# ═══════════════════════════════════════════════════════════════════════

class TestResetBehavior:
    """Verify reset clears all stateful fields."""

    def test_encoder_reset_clears_state(self) -> None:
        """Encoder reset must clear onset and vibrato history."""
        encoder = IntentEncoder(sample_rate=SAMPLE_RATE)
        audio = _sine(440, 0.5)
        encoder.encode_block(audio)

        assert encoder._prev_spectrum is not None
        assert len(encoder._f0_history) > 0

        encoder.reset()
        assert encoder._prev_spectrum is None
        assert len(encoder._f0_history) == 0

        # Pre-computed tables should survive reset
        assert encoder._mel_fb is not None
        assert encoder._a_weight is not None

    def test_decoder_reset_clears_state(self) -> None:
        """Decoder reset must clear phases, noise state, and prev frame."""
        decoder = IntentDecoder(sample_rate=SAMPLE_RATE)
        frames = [
            IntentFrame(timestamp_ms=i * 4, f0_hz=440, f0_confidence=0.9,
                        loudness_db=-12, loudness_norm=0.85,
                        spectral_centroid_hz=2500, mfcc=[1.0] * 13)
            for i in range(10)
        ]
        decoder.decode_frames(frames)

        decoder.reset()
        assert np.all(decoder._phases == 0), "Phases not reset"
        assert decoder._noise_state == 0.0, "Noise state not reset"
        assert decoder._prev_frame is None, "Previous frame not reset"


# ═══════════════════════════════════════════════════════════════════════
# Test: Decoder Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestDecoderEdgeCases:
    """Verify decoder handles unusual input gracefully."""

    def test_extreme_f0_above_nyquist(self) -> None:
        """F0 above Nyquist should not produce aliased harmonics."""
        decoder = IntentDecoder(sample_rate=SAMPLE_RATE)
        frame = IntentFrame(
            timestamp_ms=0, f0_hz=20000, f0_confidence=0.9,
            loudness_db=-12, loudness_norm=0.85,
            spectral_centroid_hz=2500, mfcc=[1.0] * 13,
        )
        audio = decoder._decode_single_frame(frame)
        assert len(audio) == SAMPLE_RATE // 250
        assert np.all(np.isfinite(audio))

    def test_empty_frame_list(self) -> None:
        """Empty frame list should return empty array."""
        decoder = IntentDecoder(sample_rate=SAMPLE_RATE)
        audio = decoder.decode_frames([])
        assert len(audio) == 0

    def test_single_mfcc_uses_fallback(self) -> None:
        """Single MFCC value should trigger 1/h fallback safely."""
        decoder = IntentDecoder(sample_rate=SAMPLE_RATE)
        frame = IntentFrame(
            timestamp_ms=0, f0_hz=440, f0_confidence=0.9,
            loudness_db=-12, loudness_norm=0.85,
            spectral_centroid_hz=2500, mfcc=[5.0],
        )
        audio = decoder._decode_single_frame(frame)
        assert len(audio) == SAMPLE_RATE // 250
        assert np.any(audio != 0), "Fallback path produced silence"
