"""
realtime_capture.py — Record from Mic and Process Through Claudio

Records a short clip from your microphone, then processes it through
the Claudio HRTF engine at every spatial position. Outputs before/after
WAV files and launches the A/B player in your browser.

This gives you REAL audio test material processed through the engine.

Usage:
    cd claudio && .venv/bin/python -m claudio.realtime_capture
    # Records 5 seconds from your mic by default
    # Or: .venv/bin/python -m claudio.realtime_capture --duration 10
"""
from __future__ import annotations

import os
import sys
import textwrap

import numpy as np
import sounddevice as sd

from claudio.audio_demo import process_through_claudio, write_wav_mono, write_wav_stereo

SAMPLE_RATE = 48000
OUTPUT_DIR = "demo_output/mic_captures"

POSITIONS = {
    "center":     np.array([0.0, 0.0, -2.0]),
    "left_30":    np.array([-1.0, 0.0, -1.73]),
    "left_45":    np.array([-1.4, 0.0, -1.4]),
    "left_90":    np.array([-2.0, 0.0, 0.0]),
    "right_30":   np.array([1.0, 0.0, -1.73]),
    "right_45":   np.array([1.4, 0.0, -1.4]),
    "right_90":   np.array([2.0, 0.0, 0.0]),
    "behind":     np.array([0.0, 0.0, 2.0]),
    "above":      np.array([0.0, 2.0, -1.0]),
}


def record_from_mic(duration_s: float = 5.0) -> np.ndarray:
    """Record audio from the default input device."""
    print(f"\n  🎙️  Recording {duration_s:.1f} seconds from microphone...")
    print("  Speak, play music, or make sounds now!")
    print()

    # Show countdown
    audio = sd.rec(
        int(duration_s * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    for i in range(int(duration_s)):
        remaining = duration_s - i
        bars = "█" * int(remaining * 4)
        print(f"\r  Recording: {remaining:.0f}s remaining {bars}    ", end="", flush=True)
        sd.sleep(1000)
    sd.wait()
    print(f"\r  ✅ Recorded {duration_s:.1f}s ({len(audio)} samples)           ")

    mono = audio[:, 0]
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    print(f"  Peak level: {peak:.4f}  RMS: {rms:.4f} ({20*np.log10(rms+1e-10):.1f} dB)")

    if peak < 0.01:
        print("  ⚠️  Very low signal — make sure your mic is working!")

    return mono


def process_all_positions(mono: np.ndarray, tag: str = "mic") -> list[dict]:
    """Process mic audio through all spatial positions."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save dry original
    dry_path = os.path.join(OUTPUT_DIR, f"{tag}_dry.wav")
    write_wav_mono(dry_path, mono, SAMPLE_RATE)
    print(f"  ✓ {dry_path}")

    results = []
    for pos_name, pos in POSITIONS.items():
        out_l, out_r = process_through_claudio(mono, pos, label=f"{tag}_{pos_name}")
        wet_path = os.path.join(OUTPUT_DIR, f"{tag}_{pos_name}_binaural.wav")
        write_wav_stereo(wet_path, out_l, out_r, SAMPLE_RATE)
        print(f"  ✓ {wet_path}")
        results.append({
            "position": pos_name,
            "dry_path": dry_path,
            "wet_path": wet_path,
        })

    return results


def generate_comparison_html(results: list[dict], tag: str = "mic") -> str:
    """Generate an HTML A/B comparison page for mic captures."""
    cards = ""
    for r in results:
        dry_file = os.path.basename(r["dry_path"])
        wet_file = os.path.basename(r["wet_path"])
        pos_label = r["position"].replace("_", " ").title()
        cards += f"""
        <div class="card">
          <h3>🎙️ Mic → {pos_label}</h3>
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
    <title>Claudio — Live Mic Capture A/B</title>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700;900&display=swap');
      :root {{
        --bg: #0a0a0f; --card: #12121a; --border: #1e1e2e;
        --accent: #6366f1; --accent2: #a855f7;
        --dry: #f59e0b; --wet: #22d3ee;
        --text: #e2e8f0; --muted: #64748b;
      }}
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      body {{
        font-family: 'Inter', sans-serif;
        background: var(--bg); color: var(--text);
        min-height: 100vh; padding: 2rem;
      }}
      .hero {{ text-align: center; padding: 2rem 1rem; }}
      .hero h1 {{
        font-size: 2.5rem; font-weight: 900;
        background: linear-gradient(135deg, #ef4444, #f59e0b);
        background-clip: text; -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }}
      .hero p {{ color: var(--muted); margin-top: 0.5rem; }}
      .tip {{
        background: rgba(239,68,68,0.1); border: 1px solid #ef4444;
        border-radius: 12px; padding: 1rem 1.5rem;
        margin: 1.5rem auto; max-width: 600px;
        font-size: 0.9rem; color: #fca5a5;
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
        gap: 1.2rem; max-width: 1200px; margin: 0 auto;
      }}
      .card {{
        background: var(--card); border: 1px solid var(--border);
        border-radius: 16px; padding: 1.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
      }}
      .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(239,68,68,0.15);
      }}
      .card h3 {{ font-weight: 600; font-size: 1rem; margin-bottom: 1rem; }}
      .ab-row {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
      .ab-col {{ flex: 1; min-width: 180px; }}
      .label {{
        display: inline-block; font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em;
        padding: 0.25rem 0.6rem; border-radius: 6px; margin-bottom: 0.5rem;
      }}
      .label.dry {{ background: rgba(245,158,11,0.15); color: var(--dry); }}
      .label.wet {{ background: rgba(34,211,238,0.15); color: var(--wet); }}
      audio {{ width: 100%; height: 36px; border-radius: 8px; }}
      .footer {{
        text-align: center; padding: 3rem 0 1rem;
        color: var(--muted); font-size: 0.8rem;
      }}
    </style>
    </head>
    <body>
    <div class="hero">
      <h1>🎙️ Live Mic Capture — A/B Test</h1>
      <p>Your real audio processed through Claudio at 9 spatial positions</p>
      <div class="tip">
        🎧 <strong>Use headphones</strong> for the full binaural effect.
        Compare your original recording against Claudio's spatial rendering.
      </div>
    </div>
    <div class="grid">
      {cards}
    </div>
    <div class="footer">
      Claudio Spatial Audio Engine — Real Microphone Capture Test
    </div>
    </body>
    </html>
    """)

    html_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  ✓ {html_path}")
    return html_path


def main() -> None:
    duration = 5.0
    if len(sys.argv) > 1 and sys.argv[1] == "--duration":
        duration = float(sys.argv[2])

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CLAUDIO MIC CAPTURE & PROCESS                          ║")
    print("║  Record → Process → Compare                             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Show devices
    devices = sd.query_devices()
    in_dev = sd.default.device[0]
    out_dev = sd.default.device[1]
    print(f"\n  Input:  [{in_dev}] {devices[in_dev]['name']}")
    print(f"  Output: [{out_dev}] {devices[out_dev]['name']}")

    mono = record_from_mic(duration)

    print("\n  Processing through Claudio at 9 positions...")
    results = process_all_positions(mono)

    print("\n  Generating A/B comparison player...")
    html_path = generate_comparison_html(results)

    print("\n  ✅ Done! Open in browser:")
    print(f"     file://{os.path.abspath(html_path)}")
    print("\n  Or start the server:")
    print("     .venv/bin/python -m claudio.audio_demo_server")
    print("     → http://localhost:8787/mic_captures/")


if __name__ == "__main__":
    main()
