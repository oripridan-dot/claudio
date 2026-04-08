"""
realtime_hifi_cli.py — CLI Entry Point for the Hi-Fi Real-Time Audio Demo

Interactive mode selector and audio stream manager for the HiFi processor.
Extracted from realtime_hifi.py for 300-line compliance.
"""
from __future__ import annotations

import sys

import numpy as np
import sounddevice as sd

from claudio.realtime_hifi import BLOCK_SIZE, HiFiProcessor


def main() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CLAUDIO HI-FI REAL-TIME AUDIO                             ║")
    print("║  Clean • High-Fidelity • Zero added latency                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    devices = sd.query_devices()
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]

    print("\n  Devices:")
    for i, d in enumerate(devices):
        tags = []
        if i == default_in:
            tags.append("IN")
        if i == default_out:
            tags.append("OUT")
        tag = f" ← {'+'.join(tags)}" if tags else ""
        ch = f"{d['max_input_channels']}in/{d['max_output_channels']}out"
        print(f"    [{i}] {d['name']:35s} {ch:10s} {int(d['default_samplerate'])}Hz{tag}")

    out_dev = devices[default_out]
    in_dev = devices[default_in]
    out_ch = min(out_dev["max_output_channels"], 2)
    out_sr = int(out_dev["default_samplerate"])

    # For BT: use the output device's native sample rate
    sample_rate = out_sr

    bt_latency_ms = 0.0
    if "Bose" in out_dev["name"] or "AirPods" in out_dev["name"] or "Bluetooth" in out_dev["name"].lower():
        bt_latency_ms = 120.0
        print(f"\n  🔵 Bluetooth detected: ~{bt_latency_ms:.0f}ms codec latency (hardware-fixed)")
        print(f"     Claudio adds < 1ms on top. Total ≈ {bt_latency_ms + BLOCK_SIZE / sample_rate * 1000:.0f}ms.")

    buffer_ms = BLOCK_SIZE / sample_rate * 1000
    print(f"\n  Config: {sample_rate}Hz / {BLOCK_SIZE} samples ({buffer_ms:.1f}ms) / {out_ch}ch")
    print(f"  Output: [{default_out}] {out_dev['name']}")
    print(f"  Input:  [{default_in}] {in_dev['name']}")

    processor = HiFiProcessor(sample_rate)

    print("\n  ┌──────────────────────────────────────────┐")
    print("  │  Modes:                                  │")
    print("  │    c  CLEAN    — pure passthrough         │")
    print("  │    e  ENHANCED — dynamics + presence      │")
    print("  │    x  CRYSTAL  — full hi-fi + stereo      │")
    print("  │                                          │")
    print("  │  Controls:                               │")
    print("  │    +/-  Output volume                    │")
    print("  │    i/o  Input / output gain              │")
    print("  │    s    Stats       q  Quit              │")
    print("  └──────────────────────────────────────────┘")
    print("\n  Mode: CLEAN | 🎧 Put on headphones and speak...\n")

    def callback(indata, outdata, frames, _time_info, status):
        if status:
            if status.input_overflow or status.output_underflow:
                processor._overruns += 1
            print(f"  ⚠ {status}", file=sys.stderr)

        mono = indata[:, 0].astype(np.float32)
        result = processor.process_block(mono)

        if out_ch >= 2:
            outdata[:] = result[:frames]
        else:
            outdata[:, 0] = (result[:frames, 0] + result[:frames, 1]) * 0.5

    running = True
    try:
        with sd.Stream(
            samplerate=sample_rate,
            blocksize=BLOCK_SIZE,
            device=(default_in, default_out),
            channels=(1, out_ch),
            dtype="float32",
            callback=callback,
            latency="low",
        ):
            while running:
                try:
                    key = input().strip().lower()
                except EOFError:
                    break

                if key == "q":
                    running = False
                elif key == "c":
                    processor._mode = "CLEAN"
                    print("  → CLEAN mode (pure passthrough)")
                elif key == "e":
                    processor._mode = "ENHANCED"
                    print("  → ENHANCED mode (dynamics + presence)")
                elif key == "x":
                    processor._mode = "CRYSTAL"
                    print("  → CRYSTAL mode (full hi-fi + stereo width)")
                elif key == "+":
                    processor._output_gain = min(processor._output_gain * 1.5, 30.0)
                    print(f"  Volume UP → {processor._output_gain:.1f}x")
                elif key == "-":
                    processor._output_gain = max(processor._output_gain / 1.5, 0.5)
                    print(f"  Volume DOWN → {processor._output_gain:.1f}x")
                elif key == "i":
                    processor._input_gain = min(processor._input_gain * 1.5, 50.0)
                    print(f"  Input gain UP → {processor._input_gain:.1f}x")
                elif key == "o":
                    processor._input_gain = max(processor._input_gain / 1.5, 1.0)
                    print(f"  Input gain DOWN → {processor._input_gain:.1f}x")
                elif key == "s":
                    _print_stats(processor, sample_rate, buffer_ms, bt_latency_ms)

    except KeyboardInterrupt:
        pass

    print("\n  Done.")


def _print_stats(
    processor: HiFiProcessor,
    sample_rate: int,
    buffer_ms: float,
    bt_latency_ms: float,
) -> None:
    """Display real-time performance statistics."""
    with processor._lock:
        times = processor._render_times.copy()
    if times:
        import statistics
        mean_us = statistics.mean(times)
        p99_us = sorted(times)[int(0.99 * len(times))]
        mx = max(times)
        deadline_us = (BLOCK_SIZE / sample_rate) * 1e6
        headroom = (1 - mean_us / deadline_us) * 100

        print("\n  ┌── Performance ─────────────────────────────────┐")
        print(f"  │ Mode:        {processor._mode:10s}                     │")
        print(f"  │ Process:     {mean_us:>6.1f}µs mean / {mx:>6.1f}µs max     │")
        print(f"  │ P99:         {p99_us:>6.1f}µs                          │")
        print(f"  │ Deadline:    {deadline_us:>6.0f}µs ({buffer_ms:.1f}ms)             │")
        print(f"  │ Headroom:    {headroom:>5.1f}%                           │")
        if bt_latency_ms > 0:
            total = bt_latency_ms + buffer_ms + mean_us / 1000
            print(f"  │ BT latency:  ~{bt_latency_ms:.0f}ms (codec)                   │")
            print(f"  │ Total E2E:   ~{total:.1f}ms                          │")
            print(f"  │ Claudio add: {mean_us/1000:.3f}ms ({mean_us:.0f}µs)              │")
        print(f"  │ Peak in:     {processor._peak_in:>7.4f}                        │")
        print(f"  │ Peak out:    {processor._peak_out:>7.4f}                        │")
        print(f"  │ Blocks:      {processor._block_count:>7d}                        │")
        print(f"  │ Overruns:    {processor._overruns:>7d}                        │")
        print(f"  │ Gain:        in={processor._input_gain:.1f}x out={processor._output_gain:.1f}x              │")
        print("  └─────────────────────────────────────────────────┘")
    else:
        print("  No data yet — speak or play audio.")


if __name__ == "__main__":
    main()
