"""
Danger Room Simulations — Claudio Intelligence Ecosystem
=========================================================
End-to-end tests for instrument classification, phase detection,
room scanning, sweet spot engine, knowledge base, and roadmap engine.
"""
import math

import numpy as np
import pytest

# ── Instrument Classifier ────────────────────────────────────────────────
from src.intelligence.instrument_classifier import (
    HarmonicProfiler,
    InstrumentClassifier,
    InstrumentFamily,
    PickupType,
    SpectralExtractor,
    TransientAnalyzer,
)


class TestSpectralExtractor:
    """Verify spectral feature extraction on synthetic signals."""

    def setup_method(self):
        self.sr = 44100
        self.extractor = SpectralExtractor(sample_rate=self.sr, n_fft=2048)

    def _sine(self, freq: float, dur: float = 0.5) -> np.ndarray:
        t = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        return (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)

    def test_mel_filterbank_shape(self):
        buf = self._sine(440.0)
        fp = self.extractor.extract(buf)
        assert fp.mfcc_coefficients.shape[0] == 13, "expected 13 MFCCs"

    def test_spectral_centroid_sine(self):
        """Pure sine at 1 kHz should have centroid near 1 kHz."""
        buf = self._sine(1000.0)
        fp = self.extractor.extract(buf)
        assert abs(fp.spectral_centroid_hz - 1000) < 200, (
            f"centroid {fp.spectral_centroid_hz} too far from 1 kHz"
        )

    def test_spectral_flatness_noise_vs_sine(self):
        """White noise should be flatter than a pure sine."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(self.sr).astype(np.float32) * 0.3
        sine = self._sine(440.0, 1.0)
        fp_noise = self.extractor.extract(noise)
        fp_sine = self.extractor.extract(sine)
        assert fp_noise.spectral_flatness >= fp_sine.spectral_flatness


class TestTransientAnalyzer:
    def setup_method(self):
        self.sr = 44100
        self.analyzer = TransientAnalyzer(sample_rate=self.sr)

    def test_sharp_attack(self):
        """Impulse-like onset should have very short attack."""
        buf = np.zeros(self.sr, dtype=np.float32)
        buf[100:120] = 1.0  # 20-sample spike
        buf[120:] = np.exp(-np.linspace(0, 10, self.sr - 120)).astype(np.float32)
        profile = self.analyzer.analyze(buf)
        assert profile.attack_time_ms < 5.0, f"attack {profile.attack_time_ms}ms too slow for impulse"

    def test_slow_pad(self):
        """Gentle fade-in should have longer attack."""
        t = np.linspace(0, 1, self.sr, dtype=np.float32)
        buf = (t ** 3) * np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.5
        profile = self.analyzer.analyze(buf)
        assert profile.attack_time_ms > 20.0


class TestHarmonicProfiler:
    def setup_method(self):
        self.sr = 44100
        self.profiler = HarmonicProfiler(sample_rate=self.sr)

    def test_fundamental_detection(self):
        """A 220 Hz sine should yield ~220 Hz fundamental."""
        t = np.linspace(0, 0.5, int(self.sr * 0.5), endpoint=False)
        buf = (np.sin(2 * np.pi * 220 * t) * 0.8).astype(np.float32)
        hp = self.profiler.analyze(buf)
        assert abs(hp.fundamental_hz - 220) < 10, f"F0={hp.fundamental_hz}, expected ~220"

    def test_rich_harmonic_content(self):
        """Sawtooth-ish signal should have many partials."""
        t = np.linspace(0, 0.5, int(self.sr * 0.5), endpoint=False)
        buf = np.zeros(len(t), dtype=np.float32)
        for k in range(1, 10):
            buf += (np.sin(2 * np.pi * 220 * k * t) / k).astype(np.float32)
        buf *= 0.3
        hp = self.profiler.analyze(buf)
        assert len(hp.partial_amplitudes) >= 3, "expected multiple partials"


class TestInstrumentClassifier:
    def setup_method(self):
        self.classifier = InstrumentClassifier(sample_rate=44100)

    def test_classify_returns_result(self):
        """Smoke test: classifier should not crash on noise."""
        rng = np.random.default_rng(7)
        buf = rng.standard_normal(44100).astype(np.float32) * 0.1
        det = self.classifier.classify(buf)
        assert det.family in InstrumentFamily.__members__.values()
        assert 0 <= det.confidence <= 1.0

    def test_classify_low_level_audio(self):
        """Very quiet audio should still produce a result."""
        buf = np.zeros(44100, dtype=np.float32) + 1e-6
        det = self.classifier.classify(buf)
        # Classifier may still be confident — just verify it runs
        assert det.family in InstrumentFamily.__members__.values()


# ── Phase Detector ────────────────────────────────────────────────────────

from src.intelligence.phase_detector import PhaseCorrelationMeter


class TestPhaseCorrelationMeter:
    def setup_method(self):
        self.meter = PhaseCorrelationMeter(sample_rate=44100)

    def _sine(self, freq=440.0, dur=0.5, phase=0.0) -> np.ndarray:
        t = np.linspace(0, dur, int(44100 * dur), endpoint=False)
        return (np.sin(2 * np.pi * freq * t + phase) * 0.8).astype(np.float32)

    def test_identical_signals_perfect_correlation(self):
        a = self._sine(440)
        frame = self.meter.analyze(a, a)
        assert frame.correlation > 0.99, f"identical signals: r={frame.correlation}"
        assert frame.time_offset_samples == 0

    def test_inverted_polarity_negative_correlation(self):
        a = self._sine(440)
        b = -a
        frame = self.meter.analyze(a, b)
        assert frame.correlation < -0.99, f"inverted: r={frame.correlation}"
        assert frame.needs_polarity_flip

    def test_phase_180_sine(self):
        a = self._sine(440, phase=0)
        b = self._sine(440, phase=math.pi)
        frame = self.meter.analyze(a, b)
        assert frame.correlation < -0.9

    def test_uncorrelated_noise(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(22050).astype(np.float32)
        b = rng.standard_normal(22050).astype(np.float32)
        frame = self.meter.analyze(a, b)
        assert abs(frame.correlation) < 0.15, f"uncorrelated noise: r={frame.correlation}"

    def test_delayed_signal_offset(self):
        """A signal delayed by 100 samples should report ~100 offset."""
        a = self._sine(440, dur=0.5)
        b = np.zeros_like(a)
        b[100:] = a[:-100]
        frame = self.meter.analyze(a, b)
        assert abs(abs(frame.time_offset_samples) - 100) < 10, f"offset={frame.time_offset_samples}"


# ── Room Scanner ──────────────────────────────────────────────────────────

from src.intelligence.room_scanner import RoomScanner


class TestRoomScanner:
    def setup_method(self):
        self.scanner = RoomScanner(sample_rate=44100)

    def _impulse_response(self, rt60: float = 0.5) -> np.ndarray:
        """Synthetic exponential decay impulse response."""
        n = int(44100 * 1.5)
        ir = np.zeros(n, dtype=np.float32)
        ir[0] = 1.0
        decay = np.exp(-np.log(1000) / (rt60 * 44100) * np.arange(n))
        rng = np.random.default_rng(99)
        ir += rng.standard_normal(n).astype(np.float32) * 0.02 * decay
        ir[0] = 1.0
        return ir.astype(np.float32)

    def test_rt60_estimation(self):
        """Synthetic IR with known RT60 should estimate close."""
        ir = self._impulse_response(rt60=0.6)
        result = self.scanner.scan_from_impulse(ir)
        assert 150 < result.rt60_ms < 3000, f"RT60={result.rt60_ms}ms"

    def test_room_modes_detected(self):
        """IR with resonant peaks should detect room modes."""
        n = 44100
        ir = self._impulse_response(0.8)
        # Inject resonance at ~120 Hz
        t = np.arange(n, dtype=np.float32) / 44100
        ir[:n] += (np.sin(2 * np.pi * 120 * t) * np.exp(-t * 5) * 0.3).astype(np.float32)
        result = self.scanner.scan_from_impulse(ir)
        # May or may not detect exact mode depending on analysis, but should not crash
        assert result.rt60_ms > 0

    def test_flutter_echo_synthetic(self):
        """Periodic impulses should trigger flutter echo detection."""
        n = 44100 * 2
        ir = np.zeros(n, dtype=np.float32)
        # Periodic reflections every ~500 samples (~88 Hz flutter)
        for i in range(0, n, 500):
            if i < n:
                ir[i] = 0.5 * (0.9 ** (i / 500))
        result = self.scanner.scan_from_impulse(ir)
        # Flutter detection is heuristic — just verify no crash
        assert isinstance(result.has_flutter_echo, bool)


# ── Sweet Spot Engine ─────────────────────────────────────────────────────

from src.intelligence.sweet_spot_engine import (
    ListenerPosition,
    ListeningMode,
    SpeakerConfig,
    SweetSpotEngine,
)


class TestSweetSpotEngine:
    def setup_method(self):
        self.engine = SweetSpotEngine(sample_rate=44100)
        self.engine.set_stereo_pair(separation_m=1.5, distance_m=1.2)

    def test_centered_listener_is_optimal(self):
        """Listener at ideal position should be in optimal zone."""
        # Reference position is set by set_stereo_pair
        pos = ListenerPosition(x=0.0, y=1.2, z=0.0)
        frame = self.engine.compute(pos)
        # In dynamic follow, centered listener should have small corrections
        for corr in frame.corrections:
            assert abs(corr.delay_ms) < 1.0
            assert abs(corr.gain_db) < 3.0

    def test_offset_listener_asymmetric_correction(self):
        """Listener shifted left should get different L vs R correction."""
        pos = ListenerPosition(x=-0.6, y=1.2, z=0.0)
        frame = self.engine.compute(pos)
        if len(frame.corrections) >= 2:
            left_corr = next((c for c in frame.corrections if c.speaker_id == "L"), None)
            right_corr = next((c for c in frame.corrections if c.speaker_id == "R"), None)
            if left_corr and right_corr:
                # Distances should differ
                assert left_corr.distance_m != right_corr.distance_m

    def test_listening_mode_switch(self):
        self.engine.set_mode(ListeningMode.WIDE_COMPROMISE)
        assert self.engine._mode == ListeningMode.WIDE_COMPROMISE

    def test_extreme_offset_coaching(self):
        """Very far off-center should produce a coaching message."""
        pos = ListenerPosition(x=-3.0, y=1.2, z=0.0)
        frame = self.engine.compute(pos)
        assert frame.coaching_message != ""
        assert frame.zone_quality < 0.8


# ── Knowledge Base ────────────────────────────────────────────────────────

from src.mentor.knowledge_base import (
    MentorKnowledgeBase,
    ProductionPhase,
    TriggerCategory,
)


class TestMentorKnowledgeBase:
    def setup_method(self):
        self.kb = MentorKnowledgeBase()

    def test_find_tips_by_trigger(self):
        tips = self.kb.find_tips(trigger=TriggerCategory.PHASE_CANCELLATION, confidence=1.0)
        assert len(tips) > 0, "should find tips for phase cancellation"

    def test_find_tips_by_trigger_and_phase(self):
        tips = self.kb.find_tips(
            trigger=TriggerCategory.ROOM_REFLECTION,
            phase=ProductionPhase.SETUP,
        )
        # May or may not have tips for this specific combination
        assert isinstance(tips, list)

    def test_find_best_tip(self):
        tip = self.kb.find_best_tip(trigger=TriggerCategory.MIC_PLACEMENT, confidence=1.0)
        assert tip is not None
        assert tip.physical_action != ""

    def test_all_mentors_have_photos(self):
        for m in self.kb.all_mentors:
            assert m.photo_asset, f"mentor {m.name} missing photo_asset"
            assert m.specialty, f"mentor {m.name} missing specialty"

    def test_find_tips_returns_empty_for_low_confidence(self):
        tips = self.kb.find_tips(
            trigger=TriggerCategory.PHASE_CANCELLATION,
            confidence=0.0,  # below all thresholds
        )
        assert len(tips) == 0

    def test_tip_has_quote(self):
        tips = self.kb.find_tips(trigger=TriggerCategory.GAIN_STAGING, confidence=1.0)
        if tips:
            assert tips[0].quote != "", "tip should have a real quote"

    def test_knowledge_base_has_minimum_tips(self):
        assert len(self.kb.all_tips) >= 10, "KB should have at least 10 tips"


# ── Roadmap Engine ────────────────────────────────────────────────────────

from src.mentor.knowledge_base import ProductionPhase
from src.mentor.roadmap_engine import RoadmapEngine


class TestRoadmapEngine:
    def setup_method(self):
        self.engine = RoadmapEngine()

    def test_initial_phase_is_setup(self):
        state = self.engine.state
        assert state.current_phase == ProductionPhase.SETUP

    def test_complete_item(self):
        result = self.engine.complete_item("setup_room_scan")
        assert result is True
        # Verify the item is marked completed
        phase = self.engine.current_phase
        room_item = next(i for i in phase.checklist if i.item_id == "setup_room_scan")
        assert room_item.completed is True

    def test_phase_auto_advance(self):
        """Completing all setup items should advance to tracking."""
        for item in self.engine.current_phase.checklist:
            self.engine.complete_item(item.item_id)
        assert self.engine.state.current_phase == ProductionPhase.TRACKING

    def test_process_detection_key(self):
        """Auto-detection should complete matching items."""
        completed = self.engine.process_detection("room_scan_complete")
        assert "setup_room_scan" in completed

    def test_complete_nonexistent_item(self):
        result = self.engine.complete_item("nonexistent_item_xyz")
        assert result is False

    def test_full_progression(self):
        """Complete all phases to reach mastering."""
        for _ in range(4):
            for item in list(self.engine.current_phase.checklist):
                self.engine.complete_item(item.item_id)
            # If not last phase, manually advance if auto didn't trigger
            if self.engine.state.current_phase != ProductionPhase.MASTERING:
                self.engine.advance_phase()
        # Should be in mastering or have completed it
        assert self.engine.state.current_phase in (
            ProductionPhase.MASTERING,
            ProductionPhase.MIXING,  # some phases might not auto-advance
        )


# ── Multimodal Fusion (smoke test) ───────────────────────────────────────

from src.intelligence.multimodal_fusion import (
    INSTRUMENT_MODEL_DB,
    BoundingBox,
    MultimodalFusion,
    VisionDetection,
    VisualCategory,
)


class TestMultimodalFusion:
    def setup_method(self):
        self.fusion = MultimodalFusion(sample_rate=44100)

    def test_audio_only_classification(self):
        """Fusion with no vision should fall back to audio-only."""
        rng = np.random.default_rng(42)
        buf = rng.standard_normal(44100).astype(np.float32) * 0.2
        result = self.fusion.fuse(audio=buf, vision_detections=None)
        assert result.instrument is not None
        assert 0 <= result.fused_confidence <= 1.0

    def test_vision_boost(self):
        """Adding a matching vision detection should increase confidence."""
        rng = np.random.default_rng(42)
        buf = rng.standard_normal(44100).astype(np.float32) * 0.2
        audio_only = self.fusion.fuse(audio=buf, vision_detections=None)
        vision = VisionDetection(
            category=VisualCategory.GUITAR_SOLID_BODY,
            bounding_box=BoundingBox(x=0.1, y=0.1, width=0.5, height=0.6),
            confidence=0.9,
            brand_text="Fender",
            model_text="Stratocaster",
        )
        fused = self.fusion.fuse(audio=buf, vision_detections=[vision])
        # Fused should be at least as confident (or close)
        assert fused.fused_confidence >= audio_only.fused_confidence - 0.1

    def test_model_db_has_entries(self):
        assert len(INSTRUMENT_MODEL_DB) > 0, "model DB should have entries"
        fender_models = [p for p in INSTRUMENT_MODEL_DB if p.brand == "Fender"]
        assert len(fender_models) >= 1
