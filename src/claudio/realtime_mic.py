"""
realtime_mic.py — Live Microphone → Claudio → Headphones

Captures audio from your Mac's microphone in real-time, processes it
through the Claudio HRTF binaural engine, and outputs the spatialized
result to your headphones — all with sub-15ms latency.

Controls (keyboard during playback):
  1-9    Select spatial position preset
  q      Quit
  s      Print real-time stats
  r      Toggle head rotation mode

Usage:
    cd claudio && .venv/bin/python -m claudio.realtime_mic
"""
from __future__ import annotations

import math
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from claudio.hrtf_engine import AudioSource, HRTFBinauralEngine
from claudio.realtime_intelligence import IntelligenceLoop
from claudio.signal_flow_config import SignalFlowConfig

# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE = 48000
BLOCK_SIZE = 512  # ~10.7ms per block at 48kHz
CHANNELS_IN = 1
CHANNELS_OUT = 2

POSITIONS = {
    "1": ("Center (front)",       np.array([0.0, 0.0, -2.0])),
    "2": ("Left 45°",             np.array([-1.4, 0.0, -1.4])),
    "3": ("Right 45°",            np.array([1.4, 0.0, -1.4])),
    "4": ("Hard Left (90°)",      np.array([-2.0, 0.0, 0.0])),
    "5": ("Hard Right (90°)",     np.array([2.0, 0.0, 0.0])),
    "6": ("Behind",               np.array([0.0, 0.0, 2.0])),
    "7": ("Above",                np.array([0.0, 2.0, -1.0])),
    "8": ("Left 30° + elevated",  np.array([-1.0, 1.0, -1.73])),
    "9": ("Orbiting (auto)",      np.array([2.0, 0.0, -1.0])),
}


class LiveProcessor:
    """Real-time mic → Claudio → headphones processor."""

    def __init__(self) -> None:
        self._cfg = SignalFlowConfig(
            capture_sample_rate=SAMPLE_RATE,
            render_sample_rate=SAMPLE_RATE,
            fft_size=BLOCK_SIZE,
            hrir_length=256,
        )
        self._engine = HRTFBinauralEngine(config=self._cfg)
        self._src = AudioSource(
            source_id="mic",
            position=POSITIONS["1"][1].copy(),
        )
        self._engine.add_source(self._src)

        self._current_pos_key = "1"
        self._rotating = False
        self._rotation_angle = 0.0
        self._running = True
        self._block_count = 0
        self._render_times: list[float] = []
        self._peak_in = 0.0
        self._peak_out = 0.0
        self._lock = threading.Lock()

        # AI Intelligence observation loop
        self._ai_enabled = True
        self._intelligence = IntelligenceLoop(
            sample_rate=SAMPLE_RATE,
            analysis_interval_s=0.5,
            analysis_window_s=1.0,
        )
        self._intelligence.start()

    def audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice for each audio block."""
        if status:
            print(f"  ⚠ {status}", file=sys.stderr)

        # Mono input
        mono = indata[:, 0].astype(np.float32)
        self._peak_in = max(self._peak_in, float(np.max(np.abs(mono))))

        # Auto-rotation mode
        if self._rotating:
            self._rotation_angle += 0.05
            radius = 2.0
            x = radius * math.sin(self._rotation_angle)
            z = -radius * math.cos(self._rotation_angle)
            self._engine.move_source("mic", np.array([x, 0.0, z]))

        # Render through Claudio
        t0 = time.perf_counter()
        frame = self._engine.render({"mic": mono})
        render_us = (time.perf_counter() - t0) * 1e6

        with self._lock:
            self._render_times.append(render_us)
            if len(self._render_times) > 500:
                self._render_times = self._render_times[-200:]

        # Output stereo
        outdata[:, 0] = frame.left[:frames]
        outdata[:, 1] = frame.right[:frames]
        self._peak_out = max(self._peak_out, float(np.max(np.abs(outdata))))
        self._block_count += 1

        # AI observation tap — copy audio to intelligence loop (non-blocking)
        if self._ai_enabled:
            self._intelligence.push_audio(mono)

    def set_position(self, key: str) -> None:
        """Switch to a preset position."""
        if key in POSITIONS:
            name, pos = POSITIONS[key]
            self._current_pos_key = key
            if key == "9":
                self._rotating = True
                print(f"  → Position: {name} (360° orbit)")
            else:
                self._rotating = False
                self._engine.move_source("mic", pos.copy())
                print(f"  → Position: {name}")

    def print_stats(self) -> None:
        """Print real-time performance stats."""
        with self._lock:
            times = self._render_times.copy()
        if not times:
            print("  No data yet")
            return

        import statistics
        mean = statistics.mean(times)
        p99 = sorted(times)[int(0.99 * len(times))]
        mx = max(times)
        deadline = (BLOCK_SIZE / SAMPLE_RATE) * 1e6
        headroom = (1 - mean / deadline) * 100

        pos_name = POSITIONS[self._current_pos_key][0]
        print("\n  ┌── Real-Time Stats ──────────────────────────────┐")
        print(f"  │ Position:    {pos_name:37s}│")
        print(f"  │ Block size:  {BLOCK_SIZE} samples ({BLOCK_SIZE/SAMPLE_RATE*1000:.1f}ms)          │")
        print(f"  │ Sample rate: {SAMPLE_RATE} Hz                        │")
        print(f"  │ Mean render: {mean:>7.1f} µs                         │")
        print(f"  │ P99 render:  {p99:>7.1f} µs                         │")
        print(f"  │ Max render:  {mx:>7.1f} µs                         │")
        print(f"  │ Deadline:    {deadline:>7.0f} µs                         │")
        print(f"  │ Headroom:    {headroom:>6.1f}%                          │")
        print(f"  │ Peak in:     {self._peak_in:>7.4f}                         │")
        print(f"  │ Peak out:    {self._peak_out:>7.4f}                         │")
        print(f"  │ Blocks:      {self._block_count:>7d}                         │")
        print("  └─────────────────────────────────────────────────┘")

        # AI Intelligence stats
        if self._ai_enabled:
            ai_stats = self._intelligence.stats
            event = self._intelligence.get_latest_event()
            print("  ┌── AI Intelligence ─────────────────────────────┐")
            print(f"  │ Status:      {'ACTIVE' if self._intelligence.is_running else 'OFF':37s}│")
            print(f"  │ Detections:  {ai_stats['total_detections']:>7d}                         │")
            print(f"  │ Instrument:  {ai_stats['last_detection']:<37s}│")
            if event and event.instrument:
                det = event.instrument
                print(f"  │ Confidence:  {det.confidence:>6.1%}                          │")
                print(f"  │ Source:      {det.classification_source:<37s}│")
                if det.pickup_type.value != 'unknown':
                    print(f"  │ Pickup:      {det.pickup_type.value:<37s}│")
                if event.coaching_hints:
                    for hint in event.coaching_hints[:2]:
                        # Truncate to fit the box
                        short = hint[:47] + '...' if len(hint) > 50 else hint
                        print(f"  │ 💡 {short:<45s}│")
                if event.mentor_tip:
                    tip = event.mentor_tip
                    print(f"  │ 🎓 {tip.mentor.name}: {tip.quote[:35]}...│")
                if event.gemini_tip:
                    gt = event.gemini_tip
                    src = f"[{gt.source}]"
                    short = gt.tip[:42] + '...' if len(gt.tip) > 45 else gt.tip
                    print(f"  │ 🤖 {src} {short:<42s}│")
            # Gemini stats
            gemini_stats = ai_stats.get("gemini")
            if gemini_stats:
                status = "ACTIVE" if gemini_stats["available"] else "fallback"
                print(f"  │ Gemini:      {status} (calls: {gemini_stats['total_calls']}, cached: {gemini_stats['total_cached']}) │")
            print("  └─────────────────────────────────────────────────┘")

    def run(self) -> None:
        """Start the live processing loop."""
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  CLAUDIO LIVE MIC PROCESSOR                             ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print("║  🎙️  Input:  Mac Microphone / Bose QC                   ║")
        print("║  🎧 Output:  Bose QC Headphones (binaural)              ║")
        print("║                                                          ║")
        print("║  Controls:                                               ║")
        print("║    1-8  Select position    9  Orbit mode                 ║")
        print("║    s    Print stats        q  Quit                       ║")
        print("║    a    Toggle AI analysis r  Head rotation              ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()
        print("  Positions:")
        for key, (name, _pos) in POSITIONS.items():
            marker = " ◀" if key == self._current_pos_key else ""
            print(f"    [{key}] {name}{marker}")
        print()
        print("  🎧 Wear headphones! Starting live processing...\n")

        try:
            with sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=(CHANNELS_IN, CHANNELS_OUT),
                dtype="float32",
                callback=self.audio_callback,
                latency="low",
            ):
                while self._running:
                    try:
                        key = input()
                        key = key.strip().lower()
                    except EOFError:
                        break

                    if key == "q":
                        self._running = False
                    elif key == "s":
                        self.print_stats()
                    elif key in POSITIONS:
                        self.set_position(key)
                    elif key == "a":
                        self._ai_enabled = not self._ai_enabled
                        if self._ai_enabled:
                            self._intelligence.start()
                            print("  → AI analysis: ON")
                        else:
                            self._intelligence.stop()
                            print("  → AI analysis: OFF")
                    else:
                        print("  Unknown command. Use 1-9, s, a, or q.")

        except KeyboardInterrupt:
            pass
        finally:
            self._intelligence.stop()
            print("\n  Shutting down...")
            self.print_stats()


def main() -> None:
    processor = LiveProcessor()
    processor.run()


if __name__ == "__main__":
    main()
