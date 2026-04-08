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
import textwrap

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

    html = textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claudio — Binaural Audio Demo</title>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700;900&display=swap');
      :root {{
        --bg:     #0a0a0f;
        --card:   #12121a;
        --border: #1e1e2e;
        --accent: #6366f1;
        --accent2:#a855f7;
        --dry:    #f59e0b;
        --wet:    #22d3ee;
        --text:   #e2e8f0;
        --muted:  #64748b;
      }}
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      body {{
        font-family: 'Inter', sans-serif;
        background: var(--bg);
        color: var(--text);
        min-height: 100vh;
        padding: 2rem;
      }}
      .hero {{
        text-align: center;
        padding: 3rem 1rem 2rem;
      }}
      .hero h1 {{
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
      }}
      .hero p {{
        font-size: 1.1rem;
        color: var(--muted);
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
      }}
      .tip {{
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.1));
        border: 1px solid var(--accent);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1.5rem auto;
        max-width: 600px;
        font-size: 0.9rem;
        color: var(--wet);
      }}
      .upload-link {{
        display: inline-block; color: var(--accent2);
        text-decoration: none; font-weight: 600;
        padding: 0.6rem 1.5rem; border: 1px solid var(--accent2);
        border-radius: 10px; transition: all 0.2s; margin-top: 1rem;
      }}
      .upload-link:hover {{ background: var(--accent2); color: #fff; }}
      .section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2.5rem 0 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid var(--accent);
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
        gap: 1.2rem;
        max-width: 1200px;
        margin: 0 auto;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
      }}
      .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99,102,241,0.15);
      }}
      .card h3 {{
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 1rem;
        color: var(--text);
      }}
      .ab-row {{
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
      }}
      .ab-col {{
        flex: 1;
        min-width: 180px;
      }}
      .label {{
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
      }}
      .label.dry {{
        background: rgba(245,158,11,0.15);
        color: var(--dry);
      }}
      .label.wet {{
        background: rgba(34,211,238,0.15);
        color: var(--wet);
      }}
      audio {{
        width: 100%;
        height: 36px;
        border-radius: 8px;
        outline: none;
      }}
      .special {{
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.08));
        border-color: var(--accent);
      }}
      .special h3 {{ color: var(--accent2); }}
      .footer {{
        text-align: center;
        padding: 3rem 0 1rem;
        color: var(--muted);
        font-size: 0.8rem;
      }}
    </style>
    </head>
    <body>
    <div class="hero">
      <h1>Claudio Binaural Demo</h1>
      <p>
        Real-time HRTF spatial audio processing. Compare dry instrument audio
        against Claudio's binaural rendering at various 3D positions.
      </p>
      <div class="tip">
        🎧 <strong>Use headphones</strong> for the full binaural effect.
        The spatial positioning is encoded in the stereo signal and
        only works with headphones, not speakers.
      </div>
      <div style="text-align:center">
        <a href="/upload" class="upload-link">📁 Upload Your Own Files</a>
      </div>
    </div>

    <div style="max-width:1200px;margin:0 auto">

    <div class="section-title">🎸 Special Demos</div>
    <div class="grid">
      <div class="card special">
        <h3>🌀 Head Rotation — Guitar orbiting around you</h3>
        <div class="ab-row">
          <div class="ab-col">
            <span class="label dry">DRY</span>
            <audio controls src="guitar_dry.wav"></audio>
          </div>
          <div class="ab-col">
            <span class="label wet">360° ROTATION</span>
            <audio controls src="guitar_head_rotation_binaural.wav"></audio>
          </div>
        </div>
      </div>
      <div class="card special">
        <h3>🎵 Full Band Mix — Spatially arranged ensemble</h3>
        <div class="ab-row">
          <div class="ab-col">
            <span class="label dry">DRY MONO MIX</span>
            <audio controls src="full_mix_dry.wav"></audio>
          </div>
          <div class="ab-col">
            <span class="label wet">BINAURAL MIX</span>
            <audio controls src="full_mix_binaural.wav"></audio>
          </div>
        </div>
      </div>
    </div>

    <div class="section-title">🎹 Instrument × Position Grid</div>
    <div class="grid">
      {cards_html}
    </div>
    </div>

    <div class="footer">
      Claudio Spatial Audio Engine — Woodworth-Schlosberg ITD · Brown-Duda ILD
      · Bilinear HRTF Interpolation · Partitioned OLA Convolution
    </div>
    </body>
    </html>
    """)

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
