"""
demo_player_template.py — HTML Template for the A/B Comparison Player

Extracted from audio_demo_runner.py for 300-line compliance.
Contains only the HTML/CSS template used by the demo player generator.
"""

from __future__ import annotations

import textwrap


def player_html(cards_html: str) -> str:
    """Return the full HTML for the A/B comparison player page."""
    return textwrap.dedent(f"""\
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
