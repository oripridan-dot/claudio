"""
realtime_intelligence.py — Real-Time AI Intelligence Loop

Observation thread that receives audio from the mic callback and
produces continuous coaching output.

Architecture:
  1. Audio callback copies frames to an observation ring buffer (lock-free)
  2. Intelligence thread wakes every ~500ms and processes buffered audio
  3. RunsInstrumentClassifier → feeds PerformanceCoach → feeds MentorKnowledgeBase
  4. Results are pushed to a thread-safe deque for UI consumption

This module respects Architecture Rule #3 (AI Observation Only):
  - It observes a COPY of the signal
  - It NEVER modifies the audio path
  - It adds zero latency to live processing
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .intelligence.instrument_classifier import InstrumentClassifier, InstrumentDetection
from .mentor.knowledge_base import MentorKnowledgeBase, MentorTip, TriggerCategory


@dataclass
class CoachingEvent:
    """A coaching event emitted by the intelligence loop."""
    timestamp: float                 # time.time()
    instrument: InstrumentDetection | None = None
    mentor_tip: MentorTip | None = None
    coaching_hints: list[str] = field(default_factory=list)
    detection_source: str = ""       # "heuristic", "neural", "fused"


class IntelligenceLoop:
    """
    Real-time intelligence observation thread.

    Receives audio frames via `push_audio()` from the mic callback,
    accumulates them in a ring buffer, and processes them periodically
    to generate coaching events.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        analysis_interval_s: float = 0.5,
        analysis_window_s: float = 1.0,
        neural_backend: object | None = None,
        max_events: int = 100,
    ):
        self._sr = sample_rate
        self._analysis_interval = analysis_interval_s
        self._window_samples = int(analysis_window_s * sample_rate)

        # Ring buffer for incoming audio (lock-free write via atomic index)
        self._buffer_size = self._window_samples * 4  # 4 seconds of headroom
        self._buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.Lock()

        # Intelligence engines
        self._classifier = InstrumentClassifier(
            sample_rate=sample_rate,
            neural_backend=neural_backend,
        )
        self._mentor_kb = MentorKnowledgeBase()

        # Output queue
        self._events: deque[CoachingEvent] = deque(maxlen=max_events)

        # State tracking
        self._running = False
        self._thread: threading.Thread | None = None
        self._total_frames = 0
        self._total_detections = 0
        self._last_detection: InstrumentDetection | None = None

    def push_audio(self, audio: np.ndarray) -> None:
        """
        Called from the audio callback thread.
        Copies audio into the observation ring buffer.
        This must be as fast as possible — no allocations, no locks on hot path.
        """
        n = len(audio)
        if n == 0:
            return

        # Simple ring buffer write (lock-free from producers perspective
        # since we only have one writer — the audio callback)
        pos = self._write_pos
        end = pos + n
        if end <= self._buffer_size:
            self._buffer[pos:end] = audio
        else:
            # Wrap around
            first_chunk = self._buffer_size - pos
            self._buffer[pos:] = audio[:first_chunk]
            self._buffer[:n - first_chunk] = audio[first_chunk:]
        self._write_pos = end % self._buffer_size
        self._total_frames += n

    def start(self) -> None:
        """Start the intelligence analysis thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._analysis_loop,
            name="claudio-intelligence",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the intelligence analysis thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest_event(self) -> CoachingEvent | None:
        """Get the most recent coaching event (non-blocking)."""
        return self._events[-1] if self._events else None

    def get_events(self, n: int = 10) -> list[CoachingEvent]:
        """Get the N most recent coaching events."""
        return list(self._events)[-n:]

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            "total_frames": self._total_frames,
            "total_detections": self._total_detections,
            "buffer_fill": self._write_pos / self._buffer_size,
            "events_queued": len(self._events),
            "last_detection": (
                self._last_detection.family.value
                if self._last_detection else "none"
            ),
        }

    def _analysis_loop(self) -> None:
        """Main intelligence analysis loop — runs on background thread."""
        last_analysis = 0.0

        while self._running:
            now = time.time()
            if now - last_analysis < self._analysis_interval:
                time.sleep(0.05)
                continue

            last_analysis = now

            # Extract analysis window from ring buffer
            audio = self._extract_window()
            if audio is None:
                continue

            # Skip silence
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.005:  # -46 dBFS threshold
                continue

            # Run instrument classification
            try:
                detection = self._classifier.classify(audio)
            except Exception:
                continue

            self._total_detections += 1
            self._last_detection = detection

            # Find relevant mentor tip
            tip = self._find_tip(detection, rms)

            # Create coaching event
            event = CoachingEvent(
                timestamp=now,
                instrument=detection,
                mentor_tip=tip,
                coaching_hints=detection.coaching_hints,
                detection_source=detection.classification_source,
            )
            self._events.append(event)

    def _extract_window(self) -> np.ndarray | None:
        """Extract the most recent analysis window from the ring buffer."""
        pos = self._write_pos
        if self._total_frames < self._window_samples:
            return None  # not enough audio yet

        # Read the last window_samples from the ring buffer
        start = (pos - self._window_samples) % self._buffer_size
        if start < pos:
            return self._buffer[start:pos].copy()
        else:
            # Wraps around
            first = self._buffer[start:]
            second = self._buffer[:pos]
            return np.concatenate([first, second])

    def _find_tip(self, det: InstrumentDetection, rms: float) -> MentorTip | None:
        """Find the most relevant mentor tip for the current detection."""
        from .intelligence.instrument_classifier import InstrumentFamily

        # High transient sharpness → harsh attack tip
        if (det.transient_profile
                and det.transient_profile.transient_sharpness > 0.8
                and det.family in (InstrumentFamily.GUITAR_ELECTRIC, InstrumentFamily.GUITAR_ACOUSTIC)):
            tip = self._mentor_kb.find_best_tip(
                TriggerCategory.HARSH_TRANSIENT,
                confidence=det.confidence,
            )
            if tip:
                return tip

        # Very bright tone → instrument setup tip
        if det.spectral_fingerprint and det.spectral_fingerprint.spectral_centroid_hz > 4000:
            tip = self._mentor_kb.find_best_tip(
                TriggerCategory.INSTRUMENT_SETUP,
                confidence=det.confidence,
            )
            if tip:
                return tip

        # High level → gain staging tip
        if rms > 0.7:
            tip = self._mentor_kb.find_best_tip(
                TriggerCategory.GAIN_STAGING,
                confidence=det.confidence,
            )
            if tip:
                return tip

        return None
