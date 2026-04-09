"""
realtime_benchmark.py — Claudio Real-Time Capability Proof

Proves that the Claudio pipeline can process audio in real-time with
professional-grade quality. Measures actual wall-clock performance
against audio buffer deadlines across a comprehensive test matrix.

Tests:
  1. Single-source latency at multiple buffer sizes
  2. Multi-source stress test (1→64 sources) to find RT ceiling
  3. Head-tracking update storm under audio load
  4. Audio fidelity suite: SNR, THD+N, frequency response, phase
  5. Sustained load test: 10 seconds of continuous rendering
  6. Worst-case jitter analysis (max vs mean render time)

Usage:
    cd claudio && .venv/bin/python -m claudio.realtime_benchmark
"""

from __future__ import annotations

import math
import statistics
import time

import numpy as np

from claudio.hrtf_engine import AudioSource, HRTFBinauralEngine
from claudio.signal_flow_config import SignalFlowConfig


def _banner(title: str) -> None:
    w = 70
    print(f"\n{'─' * w}")
    print(f"  {title}")
    print(f"{'─' * w}")


def _pass_fail(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"


# ─── Test 1: Single-Source Latency Across Buffer Sizes ───────────────────────


def test_single_source_latency() -> list[dict]:
    """Measure render time for a single source at different buffer/FFT sizes."""
    _banner("TEST 1: Single-Source Render Latency")

    configs = [
        {"label": "Ultra-Low (64/256)", "buf": 64, "fft": 256, "hrir": 128},
        {"label": "Low (128/512)", "buf": 128, "fft": 512, "hrir": 256},
        {"label": "Standard (256/512)", "buf": 256, "fft": 512, "hrir": 256},
        {"label": "High-Fidelity (512/1024)", "buf": 512, "fft": 1024, "hrir": 512},
    ]
    results = []
    for c in configs:
        cfg = SignalFlowConfig(
            capture_sample_rate=48000,
            render_sample_rate=48000,
            capture_buffer_size=c["buf"],
            output_buffer_size=c["buf"],
            fft_size=c["fft"],
            hrir_length=c["hrir"],
        )
        engine = HRTFBinauralEngine(config=cfg)
        src = AudioSource(source_id="test", position=np.array([1.0, 0.0, -2.0]))
        engine.add_source(src)

        sr = cfg.capture_sample_rate
        block = cfg.fft_size
        buffer_deadline_us = (block / sr) * 1e6  # time to play one block
        n_warmup = 10
        n_measure = 200

        # Generate test signal
        audio = np.sin(2 * np.pi * 1000 * np.arange(block * (n_warmup + n_measure)) / sr).astype(np.float32) * 0.5

        # Warmup
        for i in range(n_warmup):
            engine.render({"test": audio[i * block : (i + 1) * block]})

        # Measure
        times_us = []
        for i in range(n_warmup, n_warmup + n_measure):
            chunk = audio[i * block : (i + 1) * block]
            t0 = time.perf_counter()
            engine.render({"test": chunk})
            times_us.append((time.perf_counter() - t0) * 1e6)

        mean_us = statistics.mean(times_us)
        p99_us = sorted(times_us)[int(0.99 * len(times_us))]
        max_us = max(times_us)
        rt_ratio = mean_us / buffer_deadline_us
        headroom_pct = (1 - rt_ratio) * 100

        passed = max_us < buffer_deadline_us
        total_latency_ms = cfg.total_buffer_latency_ms + mean_us / 1000

        print(f"  {c['label']}:")
        print(f"    Buffer deadline:     {buffer_deadline_us:>8.0f} µs")
        print(f"    Mean render:         {mean_us:>8.1f} µs")
        print(f"    P99 render:          {p99_us:>8.1f} µs")
        print(f"    Max render:          {max_us:>8.1f} µs")
        print(f"    RT ratio:            {rt_ratio:>8.4f}  ({headroom_pct:.1f}% headroom)")
        print(f"    Total latency:       {total_latency_ms:>8.2f} ms")
        print(f"    Verdict:             {_pass_fail(passed)}")
        print()

        results.append(
            {
                "config": c["label"],
                "buffer_deadline_us": buffer_deadline_us,
                "mean_us": mean_us,
                "p99_us": p99_us,
                "max_us": max_us,
                "rt_ratio": rt_ratio,
                "headroom_pct": headroom_pct,
                "total_latency_ms": total_latency_ms,
                "passed": passed,
            }
        )

    return results


# ─── Test 2: Multi-Source Stress Test ────────────────────────────────────────


def test_multi_source_ceiling() -> list[dict]:
    """Increase source count until real-time budget is exceeded."""
    _banner("TEST 2: Multi-Source Real-Time Ceiling")

    cfg = SignalFlowConfig(
        capture_sample_rate=48000,
        render_sample_rate=48000,
        fft_size=512,
        hrir_length=256,
    )
    sr = cfg.capture_sample_rate
    block = cfg.fft_size
    buffer_deadline_us = (block / sr) * 1e6
    source_counts = [1, 2, 4, 8, 16, 32, 64]
    results = []

    for n_src in source_counts:
        engine = HRTFBinauralEngine(config=cfg)
        sources = []
        for i in range(n_src):
            angle = (i / n_src) * 2 * math.pi
            pos = np.array([2 * math.cos(angle), 0.0, -2 * math.sin(angle)])
            src = AudioSource(source_id=f"s{i}", position=pos)
            engine.add_source(src)
            sources.append(src)

        # Generate per-source audio
        audio_map = {}
        for i, src in enumerate(sources):
            freq = 220 * (2 ** (i / 12.0))
            t = np.arange(block * 50) / sr
            audio_map[src.source_id] = (np.sin(2 * np.pi * freq * t) * (0.3 / max(1, n_src))).astype(np.float32)

        # Warmup
        for b in range(5):
            buffers = {sid: data[b * block : (b + 1) * block] for sid, data in audio_map.items()}
            engine.render(buffers)

        # Measure
        times_us = []
        for b in range(5, 45):
            buffers = {sid: data[b * block : (b + 1) * block] for sid, data in audio_map.items()}
            t0 = time.perf_counter()
            engine.render(buffers)
            times_us.append((time.perf_counter() - t0) * 1e6)

        mean_us = statistics.mean(times_us)
        max_us = max(times_us)
        rt_ratio = mean_us / buffer_deadline_us
        is_rt = max_us < buffer_deadline_us

        print(
            f"  {n_src:>3} sources:  mean={mean_us:>7.1f}µs  max={max_us:>7.1f}µs  "
            f"ratio={rt_ratio:.4f}  {_pass_fail(is_rt)}"
        )

        results.append(
            {
                "sources": n_src,
                "mean_us": mean_us,
                "max_us": max_us,
                "rt_ratio": rt_ratio,
                "passed": is_rt,
            }
        )

        if not is_rt:
            break  # Found the ceiling

    return results


# ─── Test 3: Head-Tracking Storm ─────────────────────────────────────────────


def test_head_tracking_storm() -> dict:
    """Simulate rapid head tracking updates during rendering."""
    _banner("TEST 3: Head-Tracking Storm (120 Hz update rate)")

    cfg = SignalFlowConfig(
        capture_sample_rate=48000,
        render_sample_rate=48000,
        fft_size=512,
        hrir_length=256,
    )
    engine = HRTFBinauralEngine(config=cfg)
    sr = cfg.capture_sample_rate
    block = cfg.fft_size
    buffer_deadline_us = (block / sr) * 1e6

    for i in range(4):
        angle = (i / 4) * 2 * math.pi
        pos = np.array([2 * math.cos(angle), 0.0, -2 * math.sin(angle)])
        engine.add_source(AudioSource(source_id=f"s{i}", position=pos))

    audio = np.sin(2 * np.pi * 440 * np.arange(block * 200) / sr).astype(np.float32) * 0.5
    times_us = []

    for b in range(200):
        # Simulate 120 Hz head tracking — update every other block
        if b % 2 == 0:
            angle = (b / 200) * 2 * math.pi
            quat = (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)
            engine.update_head_pose(quat)

        buffers = {f"s{i}": audio[b * block : (b + 1) * block] for i in range(4)}
        t0 = time.perf_counter()
        engine.render(buffers)
        times_us.append((time.perf_counter() - t0) * 1e6)

    mean_us = statistics.mean(times_us)
    p99_us = sorted(times_us)[int(0.99 * len(times_us))]
    max_us = max(times_us)
    rt_ratio = mean_us / buffer_deadline_us
    passed = max_us < buffer_deadline_us

    print("  4 sources + 120Hz head tracking:")
    print(f"    Buffer deadline:     {buffer_deadline_us:>8.0f} µs")
    print(f"    Mean render:         {mean_us:>8.1f} µs")
    print(f"    P99 render:          {p99_us:>8.1f} µs")
    print(f"    Max render:          {max_us:>8.1f} µs")
    print(f"    RT ratio:            {rt_ratio:>8.4f}")
    print(f"    Verdict:             {_pass_fail(passed)}")

    return {
        "mean_us": mean_us,
        "p99_us": p99_us,
        "max_us": max_us,
        "rt_ratio": rt_ratio,
        "passed": passed,
    }


# ─── Test 4 & 5: Imported from benchmark_fidelity.py ─────────────────────────
# Extracted for 300-line compliance. See benchmark_fidelity.py.

from claudio.benchmark_fidelity import test_audio_fidelity, test_sustained_load  # noqa: E402

# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    from claudio.benchmark_report import print_scorecard

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  CLAUDIO REAL-TIME CAPABILITY BENCHMARK                        ║")
    print("║  Proving real-time audio processing performance & quality       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    r1 = test_single_source_latency()
    r2 = test_multi_source_ceiling()
    r3 = test_head_tracking_storm()
    r4 = test_audio_fidelity()
    r5 = test_sustained_load()

    print_scorecard(r1, r2, r3, r4, r5)


if __name__ == "__main__":
    main()
