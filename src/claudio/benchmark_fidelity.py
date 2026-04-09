"""
benchmark_fidelity.py — Audio Fidelity & Sustained Load Benchmarks

Tests 4 and 5 from the real-time benchmark suite.
Extracted from realtime_benchmark.py for 300-line compliance.
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


# ─── Test 4: Audio Fidelity Suite ────────────────────────────────────────────


def test_audio_fidelity() -> dict:
    """Measure SNR, THD+N, phase coherence, and frequency response."""
    _banner("TEST 4: Audio Fidelity Measurements")

    cfg = SignalFlowConfig(
        capture_sample_rate=48000,
        render_sample_rate=48000,
        fft_size=512,
        hrir_length=256,
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
        chunk = reference[b * block : (b + 1) * block]
        frame = engine.render({"fidelity": chunk})
        out_l_parts.append(frame.left)
        out_r_parts.append(frame.right)

    out_l = np.concatenate(out_l_parts)
    out_r = np.concatenate(out_r_parts)

    # Scale reference to match output level for fair SNR
    ref_trimmed = reference[: len(out_l)]
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
            chunk = test[b * block : (b + 1) * block]
            frame = engine2.render({"sweep": chunk})
            levels.append(float(np.sqrt(np.mean(frame.left**2))))
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
        "snr_db": snr,
        "thdn_pct": thdn,
        "coherence": coherence,
        "peak": peak,
        "flatness_db": flatness_db,
        "passed": snr_pass and thdn_pass and coherence_pass and peak_pass,
    }


# ─── Test 5: Sustained Load (10 seconds) ────────────────────────────────────


def test_sustained_load() -> dict:
    """10 seconds of continuous rendering — check for drift or degradation."""
    _banner("TEST 5: Sustained Load (10s continuous rendering)")

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
        "audio_time_s": audio_time,
        "wall_time_s": wall_time,
        "rt_factor": audio_time / wall_time,
        "total_blocks": total_blocks,
        "mean_us": mean_us,
        "p99_us": p99_us,
        "max_us": max_us,
        "jitter_ratio": jitter_ratio,
        "drift_pct": drift_pct,
        "overruns": n_overruns,
        "overrun_pct": overrun_pct,
        "passed": passed,
    }
