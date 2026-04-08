"""
realtime_test_tone.py — Headphone Output Test + Live Mic Processor

Phase 1: Plays a LOUD test tone through Claudio to verify output.
Phase 2: Live mic → Claudio → headphones with proper gain staging.

Automatically selects the best stereo output device and applies
makeup gain so output level matches what you'd expect from normal
system volume.

Usage:
    cd claudio && .venv/bin/python -m claudio.realtime_test_tone
"""
from __future__ import annotations

import math
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from claudio.hrtf_engine import AudioSource, HRTFBinauralEngine
from claudio.signal_flow_config import SignalFlowConfig

BLOCK_SIZE = 512

# Makeup gain to compensate for HRTF attenuation (proximity + air absorption)
# Without this, output is ~20dB below input level
MAKEUP_GAIN = 8.0

# Mic input gain — boosts quiet mic signals
MIC_GAIN = 12.0

POSITIONS = {
    "1": ("Center",          np.array([0.0, 0.0, -2.0])),
    "2": ("Left 45°",        np.array([-1.4, 0.0, -1.4])),
    "3": ("Right 45°",       np.array([1.4, 0.0, -1.4])),
    "4": ("Hard Left",       np.array([-2.0, 0.0, 0.0])),
    "5": ("Hard Right",      np.array([2.0, 0.0, 0.0])),
    "6": ("Behind",          np.array([0.0, 0.0, 2.0])),
    "7": ("Above",           np.array([0.0, 2.0, -1.0])),
    "9": ("Orbit",           np.array([2.0, 0.0, -1.0])),
}


def choose_devices() -> tuple[int | None, int, int, int]:
    """Let user confirm or choose I/O devices. Returns (in, out, sr, out_ch)."""
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]

    print("\n  Audio devices:")
    for i, d in enumerate(devices):
        ch = f"in={d['max_input_channels']} out={d['max_output_channels']}"
        sr = int(d["default_samplerate"])
        markers = []
        if i == default_in:
            markers.append("DEFAULT IN")
        if i == default_out:
            markers.append("DEFAULT OUT")
        tag = f" ← {', '.join(markers)}" if markers else ""
        print(f"    [{i}] {d['name']:35s} {ch:15s} {sr}Hz{tag}")

    # Use system defaults — these follow macOS Sound settings
    out_idx = default_out
    in_idx = default_in if default_in is not None else None

    out_dev = devices[out_idx]
    out_ch = min(out_dev["max_output_channels"], 2)
    out_sr = int(out_dev["default_samplerate"])

    print(f"\n  → Output: [{out_idx}] {out_dev['name']} ({out_ch}ch, {out_sr}Hz)")
    if in_idx is not None:
        print(f"  → Input:  [{in_idx}] {devices[in_idx]['name']}")

    if out_ch < 2:
        print("\n  ⚠  Mono output — spatial L/R will be summed to mono.")
        print("     For stereo binaural, switch Bose to A2DP/Music in macOS Sound.")

    # Allow override
    print("\n  Press Enter to use defaults, or type output device number: ", end="", flush=True)
    try:
        resp = input().strip()
    except EOFError:
        resp = ""

    if resp.isdigit():
        idx = int(resp)
        if 0 <= idx < len(devices) and devices[idx]["max_output_channels"] >= 1:
            out_idx = idx
            out_dev = devices[out_idx]
            out_ch = min(out_dev["max_output_channels"], 2)
            out_sr = int(out_dev["default_samplerate"])
            print(f"  → Using: [{out_idx}] {out_dev['name']} ({out_ch}ch, {out_sr}Hz)")

    return in_idx, out_idx, out_sr, out_ch



def apply_makeup_gain(left: np.ndarray, right: np.ndarray, gain: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply makeup gain with soft clipping to prevent distortion."""
    left_g = left * gain
    right_g = right * gain
    # Soft clip using tanh to prevent harsh digital clipping
    left_g = np.tanh(left_g * 0.8) / 0.8
    right_g = np.tanh(right_g * 0.8) / 0.8
    return left_g, right_g


def play_test_tone() -> None:
    """Play a spatialized test tone — no mic needed."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CLAUDIO AUDIO OUTPUT TEST                              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    in_idx, out_idx, sample_rate, out_channels = choose_devices()

    print(f"\n  Makeup gain: {MAKEUP_GAIN}x | Mic gain: {MIC_GAIN}x")

    cfg = SignalFlowConfig(
        capture_sample_rate=sample_rate,
        render_sample_rate=sample_rate,
        fft_size=BLOCK_SIZE,
        hrir_length=256,
    )
    engine = HRTFBinauralEngine(config=cfg)
    src = AudioSource(source_id="tone", position=np.array([-2.0, 0.0, 0.0]))
    engine.add_source(src)

    print("\n  🔊 Phase 1: Orbiting test tone (5 seconds)")
    print("     Should be clearly audible and moving around you.\n")

    duration = 5.0
    total_samples = int(duration * sample_rate)
    total_blocks = total_samples // BLOCK_SIZE
    output_buffer = np.zeros((total_blocks * BLOCK_SIZE, out_channels), dtype=np.float32)

    for b in range(total_blocks):
        t_start = b * BLOCK_SIZE / sample_rate
        t = np.arange(BLOCK_SIZE, dtype=np.float64) / sample_rate + t_start
        # Full amplitude tone
        tone = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)

        angle = (b / total_blocks) * 2 * math.pi
        x = 2.0 * math.sin(angle)
        z = -2.0 * math.cos(angle)
        engine.move_source("tone", np.array([x, 0.0, z]))

        frame = engine.render({"tone": tone})
        left, right = apply_makeup_gain(frame.left, frame.right, MAKEUP_GAIN)

        start = b * BLOCK_SIZE
        end = start + BLOCK_SIZE
        if out_channels >= 2:
            output_buffer[start:end, 0] = left
            output_buffer[start:end, 1] = right
        else:
            output_buffer[start:end, 0] = (left + right) * 0.5

    # Report peak level
    peak = float(np.max(np.abs(output_buffer)))
    print(f"  Peak output level: {peak:.3f} ({20*math.log10(peak+1e-10):.1f} dBFS)")

    sd.play(output_buffer, samplerate=sample_rate, device=out_idx)
    sd.wait()

    print("  ✅ Test tone complete.\n")

    # --- Phase 2: Live mic ---
    if in_idx is None:
        print("  No input device found.")
        return

    devices = sd.query_devices()
    in_dev = devices[in_idx]
    mic_sr = int(in_dev["default_samplerate"])
    print(f"  Mic: [{in_idx}] {in_dev['name']} ({mic_sr} Hz)")
    print("  Press Enter to start live mic, or 'q' to quit: ", end="", flush=True)

    try:
        resp = input().strip().lower()
    except EOFError:
        resp = "q"
    if resp == "q":
        return

    live_sr = min(mic_sr, sample_rate)
    print(f"\n  🎙️ Live mic → Claudio → speakers at {live_sr} Hz")
    print(f"  Gain: input ×{MIC_GAIN}, output ×{MAKEUP_GAIN}")
    print("  Controls: 1-7 position, 9 orbit, +/- volume, s stats, q quit\n")

    cfg2 = SignalFlowConfig(
        capture_sample_rate=live_sr, render_sample_rate=live_sr,
        fft_size=BLOCK_SIZE, hrir_length=256,
    )
    engine2 = HRTFBinauralEngine(config=cfg2)
    mic_src = AudioSource(source_id="mic", position=POSITIONS["1"][1].copy())
    engine2.add_source(mic_src)

    rotating = False
    rotation_angle = 0.0
    render_times: list[float] = []
    peak_in = [0.0]
    peak_out = [0.0]
    block_count = [0]
    output_gain = [MAKEUP_GAIN]
    input_gain = [MIC_GAIN]
    lock = threading.Lock()
    running = [True]

    def callback(indata, outdata, frames, _time_info, status):
        nonlocal rotation_angle
        if status:
            print(f"  ⚠ {status}", file=sys.stderr)

        # Apply input gain to boost quiet mic
        mono = indata[:, 0].astype(np.float32) * input_gain[0]
        peak_in[0] = max(peak_in[0], float(np.max(np.abs(mono))))

        if rotating:
            rotation_angle += 0.05
            x = 2.0 * math.sin(rotation_angle)
            z = -2.0 * math.cos(rotation_angle)
            engine2.move_source("mic", np.array([x, 0.0, z]))

        t0 = time.perf_counter()
        frame = engine2.render({"mic": mono})
        elapsed = (time.perf_counter() - t0) * 1e6

        # Apply makeup gain with soft clipping
        left, right = apply_makeup_gain(
            frame.left, frame.right, output_gain[0])

        with lock:
            render_times.append(elapsed)
            if len(render_times) > 500:
                del render_times[:300]

        if out_channels >= 2:
            outdata[:, 0] = left[:frames]
            outdata[:, 1] = right[:frames]
        else:
            outdata[:, 0] = ((left[:frames] + right[:frames]) * 0.5)

        out_peak = float(np.max(np.abs(outdata)))
        peak_out[0] = max(peak_out[0], out_peak)
        block_count[0] += 1

    try:
        with sd.Stream(
            samplerate=live_sr, blocksize=BLOCK_SIZE,
            device=(in_idx, out_idx),
            channels=(1, out_channels),
            dtype="float32", callback=callback, latency="low",
        ):
            while running[0]:
                try:
                    key = input().strip().lower()
                except EOFError:
                    break

                if key == "q":
                    running[0] = False
                elif key == "s":
                    with lock:
                        times = render_times.copy()
                    if times:
                        import statistics
                        mean = statistics.mean(times)
                        mx = max(times)
                        deadline = (BLOCK_SIZE / live_sr) * 1e6
                        print(f"  Render: {mean:.0f}µs (max {mx:.0f}µs) "
                              f"headroom={((1-mean/deadline)*100):.1f}%  "
                              f"peak_in={peak_in[0]:.4f}  peak_out={peak_out[0]:.4f}  "
                              f"gain_in={input_gain[0]:.1f}x  gain_out={output_gain[0]:.1f}x")
                elif key == "+":
                    output_gain[0] = min(output_gain[0] * 1.5, 50.0)
                    print(f"  Volume UP → output gain = {output_gain[0]:.1f}x")
                elif key == "-":
                    output_gain[0] = max(output_gain[0] / 1.5, 1.0)
                    print(f"  Volume DOWN → output gain = {output_gain[0]:.1f}x")
                elif key in POSITIONS:
                    name, pos = POSITIONS[key]
                    if key == "9":
                        rotating = True
                        print(f"  → {name} (orbiting)")
                    else:
                        rotating = False
                        engine2.move_source("mic", pos.copy())
                        print(f"  → {name}")
                else:
                    print("  Keys: 1-7 position, 9 orbit, +/- volume, s stats, q quit")
    except KeyboardInterrupt:
        pass

    print("  Done.")


if __name__ == "__main__":
    play_test_tone()
