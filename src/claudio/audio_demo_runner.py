"""
audio_demo_runner.py — Generate All Demo Audio Files + Web Player

Produces a complete before/after audio comparison:
  1. Raw instrument audio (mono WAV)
  2. Claudio-processed binaural audio (stereo WAV) at multiple positions
  3. A full-band mix with all instruments spatially placed
  4. An HTML A/B player for instant listening comparison

Usage:
    cd claudio && .venv/bin/python -m claudio.audio_demo_runner
"""
from __future__ import annotations

import math
import os

import numpy as np

from claudio.audio_demo import (
    DEMO_DIR,
    DEMO_SAMPLE_RATE,
    _bass_line,
    _guitar_phrase,
    _piano_chord_progression,
    process_through_claudio,
    synthesise_drum_pattern,
    write_wav_mono,
    write_wav_stereo,
)
from claudio.hrtf_engine import AudioSource, HRTFBinauralEngine
from claudio.signal_flow_config import SignalFlowConfig

# Spatial positions (x=right, y=up, z=behind)
POSITIONS = {
    "center":     np.array([0.0, 0.0, -2.0]),    # directly in front
    "left_45":    np.array([-1.4, 0.0, -1.4]),   # 45° left
    "right_45":   np.array([1.4, 0.0, -1.4]),    # 45° right
    "hard_left":  np.array([-2.0, 0.0, 0.0]),    # 90° left
    "hard_right": np.array([2.0, 0.0, 0.0]),     # 90° right
    "behind":     np.array([0.0, 0.0, 2.0]),      # directly behind
    "above":      np.array([0.0, 2.0, -1.0]),     # elevated
}


def generate_instrument_demos() -> list[dict]:
    """Generate before/after WAV files for each instrument."""
    os.makedirs(DEMO_DIR, exist_ok=True)
    sr = DEMO_SAMPLE_RATE
    demos: list[dict] = []

    instruments = {
        "guitar": _guitar_phrase(sr),
        "bass": _bass_line(sr),
        "piano": _piano_chord_progression(sr),
        "drums": synthesise_drum_pattern(4.0, bpm=120.0, sr=sr),
    }

    for name, audio in instruments.items():
        # Save dry mono
        dry_path = os.path.join(DEMO_DIR, f"{name}_dry.wav")
        write_wav_mono(dry_path, audio, sr)
        print(f"  ✓ {dry_path}")

        # Process at each position
        for pos_name, pos in POSITIONS.items():
            out_l, out_r = process_through_claudio(audio, pos, label=f"{name}_{pos_name}")
            wet_path = os.path.join(DEMO_DIR, f"{name}_{pos_name}_binaural.wav")
            write_wav_stereo(wet_path, out_l, out_r, sr)
            print(f"  ✓ {wet_path}")
            demos.append({"instrument": name, "position": pos_name, "dry": dry_path, "wet": wet_path})

    return demos


def generate_full_mix() -> str:
    """Generate a full-band mix with instruments spatially arranged."""
    sr = DEMO_SAMPLE_RATE
    cfg = SignalFlowConfig(
        capture_sample_rate=sr, render_sample_rate=sr,
        fft_size=512, hrir_length=256,
    )
    engine = HRTFBinauralEngine(config=cfg)

    # Arrange instruments in a realistic stage layout
    instruments = {
        "guitar":  {"audio": _guitar_phrase(sr), "pos": np.array([-1.5, 0.0, -2.0])},
        "bass":    {"audio": _bass_line(sr),     "pos": np.array([1.5, 0.0, -2.0])},
        "piano":   {"audio": _piano_chord_progression(sr), "pos": np.array([0.0, 0.3, -3.0])},
        "drums":   {"audio": synthesise_drum_pattern(4.0, bpm=120.0, sr=sr), "pos": np.array([0.0, 0.5, -4.0])},
    }

    # Find the longest audio
    max_len = max(len(v["audio"]) for v in instruments.values())
    block = cfg.fft_size

    # Add sources to engine
    for name, data in instruments.items():
        src = AudioSource(source_id=name, position=data["pos"])
        engine.add_source(src)
        # Pad audio to max_len
        padded = np.zeros(max_len, dtype=np.float32)
        padded[:len(data["audio"])] = data["audio"]
        data["padded"] = padded

    n_blocks = max_len // block
    out_l_parts, out_r_parts = [], []

    for b in range(n_blocks):
        buffers = {}
        for name, data in instruments.items():
            buffers[name] = data["padded"][b * block:(b + 1) * block]
        frame = engine.render(buffers)
        out_l_parts.append(frame.left)
        out_r_parts.append(frame.right)

    out_l = np.concatenate(out_l_parts)
    out_r = np.concatenate(out_r_parts)

    # Normalise
    peak = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-6)
    out_l *= 0.85 / peak
    out_r *= 0.85 / peak

    mix_path = os.path.join(DEMO_DIR, "full_mix_binaural.wav")
    write_wav_stereo(mix_path, out_l, out_r, sr)

    # Also create a dry mono downmix for comparison
    dry_mix = np.zeros(max_len, dtype=np.float32)
    for data in instruments.values():
        dry_mix[:len(data["audio"])] += data["audio"] * 0.25
    dry_path = os.path.join(DEMO_DIR, "full_mix_dry.wav")
    write_wav_mono(dry_path, dry_mix, sr)

    print(f"  ✓ {dry_path}")
    print(f"  ✓ {mix_path}")
    return mix_path


def generate_head_rotation_demo() -> str:
    """Guitar playing while head slowly rotates 360° — dramatic spatial demo."""
    sr = DEMO_SAMPLE_RATE
    cfg = SignalFlowConfig(
        capture_sample_rate=sr, render_sample_rate=sr,
        fft_size=512, hrir_length=256,
    )
    engine = HRTFBinauralEngine(config=cfg)
    audio = _guitar_phrase(sr)
    src = AudioSource(source_id="guitar_rot", position=np.array([2.0, 0.0, -1.0]))
    engine.add_source(src)

    block = cfg.fft_size
    n_blocks = len(audio) // block
    out_l_parts, out_r_parts = [], []

    for b in range(n_blocks):
        # Rotate head smoothly over the duration
        angle = (b / max(1, n_blocks)) * 2 * math.pi
        quat = (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)
        engine.update_head_pose(quat)

        chunk = audio[b * block:(b + 1) * block]
        frame = engine.render({"guitar_rot": chunk})
        out_l_parts.append(frame.left)
        out_r_parts.append(frame.right)

    out_l = np.concatenate(out_l_parts)
    out_r = np.concatenate(out_r_parts)
    peak = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-6)
    out_l *= 0.85 / peak
    out_r *= 0.85 / peak

    path = os.path.join(DEMO_DIR, "guitar_head_rotation_binaural.wav")
    write_wav_stereo(path, out_l, out_r, sr)
    print(f"  ✓ {path}")
    return path


def generate_html_player(demos: list[dict]) -> str:
    """Generate an HTML A/B comparison player."""
    cards_html = ""
    for d in demos:
        dry_file = os.path.basename(d["dry"])
        wet_file = os.path.basename(d["wet"])
        cards_html += f"""
        <div class="card">
          <h3>{d['instrument'].title()} — {d['position'].replace('_', ' ').title()}</h3>
          <div class="ab-row">
            <div class="ab-col">
              <span class="label dry">DRY (Original)</span>
              <audio controls src="{dry_file}"></audio>
            </div>
            <div class="ab-col">
              <span class="label wet">BINAURAL (Claudio)</span>
              <audio controls src="{wet_file}"></audio>
            </div>
          </div>
        </div>"""

    from claudio.demo_player_template import player_html
    html = player_html(cards_html)

    html_path = os.path.join(DEMO_DIR, "index.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  ✓ {html_path}")
    return html_path


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Claudio Audio Demo Generator                           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Generating instrument demos...")
    demos = generate_instrument_demos()
    print()
    print("Generating full-band binaural mix...")
    generate_full_mix()
    print()
    print("Generating head rotation demo...")
    generate_head_rotation_demo()
    print()
    print("Generating HTML A/B player...")
    html_path = generate_html_player(demos)
    print()
    print(f"✅ Done! Open {html_path} in a browser to listen.")
    print("🎧 Use headphones for the full binaural spatial effect!")


if __name__ == "__main__":
    main()
