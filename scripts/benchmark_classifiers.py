#!/usr/bin/env python3
"""
benchmark_classifiers.py — Head-to-Head Neural Audio Classifier Benchmark

Tests PANNs, CLAP, and BEATs on the same audio samples and produces a
comparison table. Supports file-based input and live microphone capture.

Usage:
  # From audio files
  python scripts/benchmark_classifiers.py --files path/to/guitar.wav path/to/drums.wav

  # From live mic (captures 3 seconds)
  python scripts/benchmark_classifiers.py --live --duration 3

  # Synthetic test tones (no external audio needed)
  python scripts/benchmark_classifiers.py --synthetic

  # Select specific backends
  python scripts/benchmark_classifiers.py --backends panns clap beats --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from claudio.intelligence.classifier_backend import (
    AudioClassifierBackend,
    BenchmarkResult,
)


def generate_synthetic_instruments(sr: int = 48_000, duration: float = 2.0) -> dict[str, np.ndarray]:
    """Generate synthetic audio approximating different instruments."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    samples: dict[str, np.ndarray] = {}

    # Electric guitar — rich harmonics, moderate attack
    f0 = 330.0  # E4
    signal = np.zeros(n)
    for h in range(1, 8):
        amp = 1.0 / (h**0.8)
        signal += amp * np.sin(2 * np.pi * f0 * h * t)
    # Add pick transient
    transient = np.exp(-t / 0.005) * np.sin(2 * np.pi * 3000 * t) * 0.3
    signal += transient
    signal *= np.exp(-t / 1.5)  # natural decay
    samples["Electric Guitar (E4, 330Hz)"] = (signal / np.max(np.abs(signal)) * 0.8).astype(np.float32)

    # Acoustic guitar — wider spectrum, body resonance
    f0 = 220.0  # A3
    signal = np.zeros(n)
    for h in range(1, 12):
        amp = 1.0 / (h**0.6)
        signal += amp * np.sin(2 * np.pi * f0 * h * t)
    # Body resonance around 200Hz
    body = np.sin(2 * np.pi * 200 * t) * 0.5 * np.exp(-t / 0.8)
    signal += body
    signal *= np.exp(-t / 2.0)
    samples["Acoustic Guitar (A3, 220Hz)"] = (signal / np.max(np.abs(signal)) * 0.8).astype(np.float32)

    # Kick drum — low thump with click
    click = np.exp(-t / 0.002) * np.sin(2 * np.pi * 4000 * t) * 0.5
    thump_freq = 60 + 180 * np.exp(-t / 0.01)  # pitch drops
    thump = np.sin(2 * np.pi * np.cumsum(thump_freq / sr)) * np.exp(-t / 0.15)
    samples["Kick Drum"] = ((click + thump) / np.max(np.abs(click + thump)) * 0.9).astype(np.float32)

    # Snare drum — bright noise + body
    noise = np.random.randn(n) * 0.4 * np.exp(-t / 0.08)
    body = np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.05) * 0.6
    snare = noise + body
    samples["Snare Drum"] = (snare / np.max(np.abs(snare)) * 0.8).astype(np.float32)

    # Piano — inharmonic partials, sharp attack, long decay
    f0 = 440.0  # A4
    signal = np.zeros(n)
    for h in range(1, 10):
        # Piano has slight inharmonicity
        partial_freq = f0 * h * (1 + 0.0005 * h**2)
        amp = 1.0 / (h**0.5)
        signal += amp * np.sin(2 * np.pi * partial_freq * t)
    signal *= np.exp(-t / 3.0)
    # Hammer attack
    hammer = np.exp(-t / 0.001) * 0.3
    signal *= 1 + hammer
    samples["Piano (A4, 440Hz)"] = (signal / np.max(np.abs(signal)) * 0.8).astype(np.float32)

    # Bass guitar — low fundamental, moderate harmonics
    f0 = 82.0  # E2
    signal = np.zeros(n)
    for h in range(1, 6):
        amp = 1.0 / (h**1.2)
        signal += amp * np.sin(2 * np.pi * f0 * h * t)
    signal *= np.exp(-t / 2.5)
    samples["Bass Guitar (E2, 82Hz)"] = (signal / np.max(np.abs(signal)) * 0.8).astype(np.float32)

    # Vocal — formant-rich, sustained
    f0 = 150.0  # male fundamental
    signal = np.zeros(n)
    formants = [270, 2300, 3000, 3300]  # F1-F4 for "a" vowel
    for formant in formants:
        bw = 80
        for h in range(1, 15):
            freq = f0 * h
            # Formant envelope
            amp = np.exp(-0.5 * ((freq - formant) / bw) ** 2)
            signal += amp * np.sin(2 * np.pi * freq * t)
    # Add vibrato
    vibrato = np.sin(2 * np.pi * 5.5 * t) * 3.0
    signal_vib = np.zeros(n)
    for h in range(1, 8):
        signal_vib += (1.0 / h) * np.sin(2 * np.pi * (f0 + vibrato) * h * t)
    signal += signal_vib * 0.3
    samples["Male Vocal (A vowel, 150Hz)"] = (signal / np.max(np.abs(signal)) * 0.7).astype(np.float32)

    return samples


def capture_live_audio(duration: float = 3.0, sr: int = 48_000) -> np.ndarray:
    """Capture audio from the default microphone."""
    import sounddevice as sd

    print(f"\n🎤 Recording {duration}s from microphone...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("✅ Recording complete")
    return audio.flatten()


def load_backends(backend_names: list[str], device: str) -> list[AudioClassifierBackend]:
    """Load specified backends."""
    backends: list[AudioClassifierBackend] = []

    for name in backend_names:
        name = name.lower()
        try:
            if name == "panns":
                from claudio.intelligence.backend_panns import PANNsBackend

                backend = PANNsBackend(device=device)
                backend.load_model()
                backends.append(backend)
            elif name == "clap":
                from claudio.intelligence.backend_clap import CLAPBackend

                backend = CLAPBackend(device=device)
                backend.load_model()
                backends.append(backend)
            elif name == "beats":
                from claudio.intelligence.backend_beats import BEATsBackend

                backend = BEATsBackend(device=device)
                backend.load_model()
                backends.append(backend)
            else:
                print(f"⚠️  Unknown backend: {name}")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            print(
                f"   Install with: pip install {'panns-inference' if name == 'panns' else 'transformers laion-clap' if name == 'clap' else 'transformers'}"
            )

    return backends


def run_benchmark(
    backends: list[AudioClassifierBackend],
    audio_samples: dict[str, np.ndarray],
    sample_rate: int = 48_000,
    n_runs: int = 5,
) -> dict[str, list[BenchmarkResult]]:
    """Run all backends on all samples and collect results."""
    all_results: dict[str, list[BenchmarkResult]] = {}

    for sample_name, audio in audio_samples.items():
        print(f"\n{'─' * 60}")
        print(f"🎵 Sample: {sample_name}")
        print(f"   Duration: {len(audio) / sample_rate:.1f}s, RMS: {np.sqrt(np.mean(audio**2)):.4f}")

        sample_results = []
        for backend in backends:
            print(f"   → Running {backend.name}...", end=" ", flush=True)
            result = backend.benchmark(audio, sample_rate, n_runs=n_runs)
            sample_results.append(result)
            print(f"✅ {result.top1_label} ({result.top1_confidence:.2%}) [{result.total_latency_ms:.1f}ms]")

        all_results[sample_name] = sample_results

    return all_results


def print_comparison_table(all_results: dict[str, list[BenchmarkResult]]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "═" * 100)
    print("NEURAL AUDIO CLASSIFIER BENCHMARK — HEAD-TO-HEAD COMPARISON")
    print("═" * 100)

    for sample_name, results in all_results.items():
        print(f"\n{'─' * 100}")
        print(f"🎵 {sample_name}")
        print(f"{'─' * 100}")
        print(f"{'Backend':<25} {'Top-1 Label':<35} {'Confidence':>10} {'Family':>20} {'Latency':>10}")
        print(f"{'─' * 25} {'─' * 35} {'─' * 10} {'─' * 20} {'─' * 10}")

        for r in results:
            print(
                f"{r.backend_name:<25} {r.top1_label:<35} "
                f"{r.top1_confidence:>9.1%} "
                f"{r.top1_family.value:>20} "
                f"{r.total_latency_ms:>8.1f}ms"
            )

            # Show top-5
            for cr in r.results[1:]:
                if cr.confidence > 0.05:
                    print(f"{'':>25} {cr.label:<35} {cr.confidence:>9.1%} {cr.family.value:>20}")

    # Summary
    print(f"\n{'═' * 100}")
    print("SUMMARY")
    print(f"{'═' * 100}")

    # Collect stats per backend
    backend_names = set()
    for results in all_results.values():
        for r in results:
            backend_names.add(r.backend_name)

    for bname in sorted(backend_names):
        latencies = []
        confidences = []
        for results in all_results.values():
            for r in results:
                if r.backend_name == bname:
                    latencies.append(r.total_latency_ms)
                    confidences.append(r.top1_confidence)

        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        print(f"  {bname:<25} Avg latency: {avg_lat:>6.1f}ms  Avg confidence: {avg_conf:>6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head neural audio classifier benchmark",
    )
    parser.add_argument("--files", nargs="+", help="Audio files to classify")
    parser.add_argument("--live", action="store_true", help="Capture from microphone")
    parser.add_argument("--duration", type=float, default=3.0, help="Live capture duration (s)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test tones")
    parser.add_argument(
        "--backends", nargs="+", default=["panns", "clap", "beats"], help="Backends to test (panns, clap, beats)"
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--sr", type=int, default=48_000, help="Sample rate")
    parser.add_argument("--runs", type=int, default=5, help="Number of inference runs per sample")
    args = parser.parse_args()

    # Collect audio samples
    audio_samples: dict[str, np.ndarray] = {}

    if args.synthetic or (not args.files and not args.live):
        print("🔧 Generating synthetic instrument samples...")
        audio_samples = generate_synthetic_instruments(args.sr)

    if args.files:
        import soundfile as sf

        for fpath in args.files:
            audio, sr = sf.read(fpath, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != args.sr:
                # Resample
                ratio = args.sr / sr
                n = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, n),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)
            audio_samples[Path(fpath).stem] = audio

    if args.live:
        live_audio = capture_live_audio(args.duration, args.sr)
        audio_samples["Live Microphone Capture"] = live_audio

    if not audio_samples:
        print("No audio samples provided. Use --synthetic, --files, or --live")
        sys.exit(1)

    print(f"\n📊 Benchmark: {len(audio_samples)} samples × {len(args.backends)} backends")
    print(f"   Device: {args.device}, Sample rate: {args.sr}Hz, Runs: {args.runs}")

    # Load backends
    backends = load_backends(args.backends, args.device)
    if not backends:
        print("❌ No backends loaded. Please install the required packages.")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(backends, audio_samples, args.sr, args.runs)

    # Print comparison
    print_comparison_table(results)

    print(f"\n✅ Benchmark complete. Tested {len(backends)} backends on {len(audio_samples)} samples.")


if __name__ == "__main__":
    main()
