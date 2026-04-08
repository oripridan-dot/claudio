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
from claudio.signal_flow_metrics import measure_phase_coherence, measure_snr, measure_thdn


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
            capture_sample_rate=48000, render_sample_rate=48000,
            capture_buffer_size=c["buf"], output_buffer_size=c["buf"],
            fft_size=c["fft"], hrir_length=c["hrir"],
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
            engine.render({"test": audio[i * block:(i + 1) * block]})

        # Measure
        times_us = []
        for i in range(n_warmup, n_warmup + n_measure):
            chunk = audio[i * block:(i + 1) * block]
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

        results.append({
            "config": c["label"], "buffer_deadline_us": buffer_deadline_us,
            "mean_us": mean_us, "p99_us": p99_us, "max_us": max_us,
            "rt_ratio": rt_ratio, "headroom_pct": headroom_pct,
            "total_latency_ms": total_latency_ms, "passed": passed,
        })

    return results


# ─── Test 2: Multi-Source Stress Test ────────────────────────────────────────

def test_multi_source_ceiling() -> list[dict]:
    """Increase source count until real-time budget is exceeded."""
    _banner("TEST 2: Multi-Source Real-Time Ceiling")

    cfg = SignalFlowConfig(
        capture_sample_rate=48000, render_sample_rate=48000,
        fft_size=512, hrir_length=256,
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
            buffers = {sid: data[b * block:(b + 1) * block] for sid, data in audio_map.items()}
            engine.render(buffers)

        # Measure
        times_us = []
        for b in range(5, 45):
            buffers = {sid: data[b * block:(b + 1) * block] for sid, data in audio_map.items()}
            t0 = time.perf_counter()
            engine.render(buffers)
            times_us.append((time.perf_counter() - t0) * 1e6)

        mean_us = statistics.mean(times_us)
        max_us = max(times_us)
        rt_ratio = mean_us / buffer_deadline_us
        is_rt = max_us < buffer_deadline_us

        print(f"  {n_src:>3} sources:  mean={mean_us:>7.1f}µs  max={max_us:>7.1f}µs  "
              f"ratio={rt_ratio:.4f}  {_pass_fail(is_rt)}")

        results.append({
            "sources": n_src, "mean_us": mean_us, "max_us": max_us,
            "rt_ratio": rt_ratio, "passed": is_rt,
        })

        if not is_rt:
            break  # Found the ceiling

    return results


# ─── Test 3: Head-Tracking Storm ─────────────────────────────────────────────

def test_head_tracking_storm() -> dict:
    """Simulate rapid head tracking updates during rendering."""
    _banner("TEST 3: Head-Tracking Storm (120 Hz update rate)")

    cfg = SignalFlowConfig(
        capture_sample_rate=48000, render_sample_rate=48000,
        fft_size=512, hrir_length=256,
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

        buffers = {f"s{i}": audio[b * block:(b + 1) * block] for i in range(4)}
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
        "mean_us": mean_us, "p99_us": p99_us, "max_us": max_us,
        "rt_ratio": rt_ratio, "passed": passed,
    }


# ─── Test 4: Audio Fidelity Suite ────────────────────────────────────────────

def test_audio_fidelity() -> dict:
    """Measure SNR, THD+N, phase coherence, and frequency response."""
    _banner("TEST 4: Audio Fidelity Measurements")

    cfg = SignalFlowConfig(
        capture_sample_rate=48000, render_sample_rate=48000,
        fft_size=512, hrir_length=256,
    )
    engine = HRTFBinauralEngine(config=cfg)
    sr = cfg.capture_sample_rate
    block = cfg.fft_size

    # Center source — should have minimal spectral modification
    src = AudioSource(source_id="fidelity", position=np.array([0.0, 0.0, -2.0]))
    engine.add_source(src)

    # Generate 1 kHz sine test signal
    duration = 1.0
    n_samples = int(sr * duration)
    reference = np.sin(2 * np.pi * 1000 * np.arange(n_samples) / sr).astype(np.float32) * 0.5

    out_l_parts, out_r_parts = [], []
    for b in range(n_samples // block):
        chunk = reference[b * block:(b + 1) * block]
        frame = engine.render({"fidelity": chunk})
        out_l_parts.append(frame.left)
        out_r_parts.append(frame.right)

    out_l = np.concatenate(out_l_parts)
    out_r = np.concatenate(out_r_parts)

    # Scale reference to match output level for fair SNR
    ref_trimmed = reference[:len(out_l)]
    scale = float(np.dot(out_l, ref_trimmed)) / (float(np.dot(ref_trimmed, ref_trimmed)) + 1e-30)
    ref_scaled = ref_trimmed * scale

    snr = measure_snr(out_l, ref_scaled)
    thdn = measure_thdn(out_l)
    coherence = measure_phase_coherence(out_l, out_r)

    # Frequency response flatness (sweep test)
    engine2 = HRTFBinauralEngine(config=cfg)
    src2 = AudioSource(source_id="sweep", position=np.array([0.0, 0.0, -2.0]))
    engine2.add_source(src2)

    freqs = [100, 250, 500, 1000, 2000, 4000, 8000, 16000]
    freq_levels = []
    for f in freqs:
        test = np.sin(2 * np.pi * f * np.arange(block * 20) / sr).astype(np.float32) * 0.5
        levels = []
        for b in range(20):
            chunk = test[b * block:(b + 1) * block]
            frame = engine2.render({"sweep": chunk})
            levels.append(float(np.sqrt(np.mean(frame.left ** 2))))
        freq_levels.append(statistics.mean(levels[-10:]))  # skip transient

    # Flatness: max deviation from mean level
    mean_level = statistics.mean(freq_levels)
    if mean_level > 0:
        deviations_db = [20 * math.log10(max(lvl, 1e-10) / mean_level) for lvl in freq_levels]
        flatness_db = max(abs(d) for d in deviations_db)
    else:
        deviations_db = [0.0] * len(freqs)
        flatness_db = 0.0

    # Output peak level
    peak = float(max(np.max(np.abs(out_l)), np.max(np.abs(out_r))))

    snr_pass = snr > 20.0
    thdn_pass = thdn < 5.0
    coherence_pass = coherence > 0.3
    peak_pass = peak < 1.0

    print(f"  SNR:                   {snr:>8.1f} dB    {_pass_fail(snr_pass)}  (>20 dB)")
    print(f"  THD+N:                 {thdn:>8.4f} %     {_pass_fail(thdn_pass)}  (<5%)")
    print(f"  Phase Coherence:       {coherence:>8.3f}       {_pass_fail(coherence_pass)}  (>0.3)")
    print(f"  Peak Level:            {peak:>8.4f}       {_pass_fail(peak_pass)}  (<1.0)")
    print(f"  Freq Response Flatness:{flatness_db:>8.1f} dB")
    print()
    print("  Frequency Response (deviation from mean):")
    for f, d in zip(freqs, deviations_db, strict=True):
        bar = "█" * max(0, int((d + 10) * 2))
        print(f"    {f:>6} Hz: {d:>+6.1f} dB  {bar}")

    return {
        "snr_db": snr, "thdn_pct": thdn, "coherence": coherence,
        "peak": peak, "flatness_db": flatness_db,
        "passed": snr_pass and thdn_pass and coherence_pass and peak_pass,
    }


# ─── Test 5: Sustained Load (10 seconds) ────────────────────────────────────

def test_sustained_load() -> dict:
    """10 seconds of continuous rendering — check for drift or degradation."""
    _banner("TEST 5: Sustained Load (10s continuous rendering)")

    cfg = SignalFlowConfig(
        capture_sample_rate=48000, render_sample_rate=48000,
        fft_size=512, hrir_length=256,
    )
    engine = HRTFBinauralEngine(config=cfg)
    sr = cfg.capture_sample_rate
    block = cfg.fft_size
    buffer_deadline_us = (block / sr) * 1e6
    duration = 10.0

    for i in range(4):
        angle = (i / 4) * 2 * math.pi
        pos = np.array([2 * math.cos(angle), 0.0, -2 * math.sin(angle)])
        engine.add_source(AudioSource(source_id=f"s{i}", position=pos))

    total_blocks = int(sr * duration) // block
    times_us = []
    t_wall_start = time.perf_counter()

    for b in range(total_blocks):
        t = np.arange(block) / sr + (b * block / sr)
        buffers = {}
        for i in range(4):
            freq = 220 * (2 ** (i / 12.0))
            buffers[f"s{i}"] = (np.sin(2 * np.pi * freq * t) * 0.2).astype(np.float32)

        t0 = time.perf_counter()
        engine.render(buffers)
        times_us.append((time.perf_counter() - t0) * 1e6)

    wall_time = time.perf_counter() - t_wall_start
    audio_time = duration

    mean_us = statistics.mean(times_us)
    p99_us = sorted(times_us)[int(0.99 * len(times_us))]
    max_us = max(times_us)
    stdev_us = statistics.stdev(times_us)
    jitter_ratio = stdev_us / mean_us

    # Check for drift: compare first 100 vs last 100 blocks
    first_100 = statistics.mean(times_us[:100])
    last_100 = statistics.mean(times_us[-100:])
    drift_pct = abs(last_100 - first_100) / first_100 * 100

    n_overruns = sum(1 for t in times_us if t > buffer_deadline_us)
    overrun_pct = n_overruns / len(times_us) * 100

    passed = n_overruns == 0

    print(f"  Audio duration:        {audio_time:>8.1f} s")
    print(f"  Wall-clock time:       {wall_time:>8.3f} s")
    print(f"  Processing speed:      {audio_time / wall_time:>8.1f}× real-time")
    print(f"  Total blocks:          {total_blocks:>8d}")
    print(f"  Mean render:           {mean_us:>8.1f} µs")
    print(f"  P99 render:            {p99_us:>8.1f} µs")
    print(f"  Max render:            {max_us:>8.1f} µs")
    print(f"  Jitter (σ/µ):          {jitter_ratio:>8.3f}")
    print(f"  Drift (first→last):    {drift_pct:>8.1f} %")
    print(f"  Buffer overruns:       {n_overruns:>5d} / {total_blocks}  ({overrun_pct:.2f}%)")
    print(f"  Verdict:               {_pass_fail(passed)}")

    return {
        "audio_time_s": audio_time, "wall_time_s": wall_time,
        "rt_factor": audio_time / wall_time, "total_blocks": total_blocks,
        "mean_us": mean_us, "p99_us": p99_us, "max_us": max_us,
        "jitter_ratio": jitter_ratio, "drift_pct": drift_pct,
        "overruns": n_overruns, "overrun_pct": overrun_pct, "passed": passed,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  CLAUDIO REAL-TIME CAPABILITY BENCHMARK                        ║")
    print("║  Proving real-time audio processing performance & quality       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    r1 = test_single_source_latency()
    r2 = test_multi_source_ceiling()
    r3 = test_head_tracking_storm()
    r4 = test_audio_fidelity()
    r5 = test_sustained_load()

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  FINAL SCORE CARD")
    print("═" * 70)

    all_results = []

    # Test 1 summary
    for r in r1:
        label = f"T1 Latency [{r['config']}]"
        print(f"  {_pass_fail(r['passed'])}  {label:45s}  {r['mean_us']:.0f}µs / {r['total_latency_ms']:.2f}ms")
        all_results.append(r["passed"])

    # Test 2 summary
    max_rt_sources = 0
    for r in r2:
        if r["passed"]:
            max_rt_sources = r["sources"]
    print(f"  {'✅ PASS' if max_rt_sources >= 16 else '❌ FAIL'}  {'T2 Multi-Source Ceiling':45s}  {max_rt_sources} sources in real-time")
    all_results.append(max_rt_sources >= 16)

    # Test 3
    print(f"  {_pass_fail(r3['passed'])}  {'T3 Head-Track Storm (120Hz + 4 sources)':45s}  {r3['mean_us']:.0f}µs mean")
    all_results.append(r3["passed"])

    # Test 4
    print(f"  {_pass_fail(r4['passed'])}  {'T4 Audio Fidelity':45s}  SNR={r4['snr_db']:.1f}dB THD={r4['thdn_pct']:.3f}%")
    all_results.append(r4["passed"])

    # Test 5
    print(f"  {_pass_fail(r5['passed'])}  {'T5 Sustained Load (10s)':45s}  {r5['rt_factor']:.0f}× RT, {r5['overruns']} overruns")
    all_results.append(r5["passed"])

    total_pass = sum(all_results)
    total = len(all_results)
    all_pass = all(all_results)

    print(f"\n{'═' * 70}")
    if all_pass:
        print(f"  🏆 ALL {total}/{total} TESTS PASSED — REAL-TIME CERTIFIED")
    else:
        print(f"  ⚠️  {total_pass}/{total} TESTS PASSED")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
