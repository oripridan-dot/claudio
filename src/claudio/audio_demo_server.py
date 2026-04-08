"""
audio_demo_server.py — Claudio Audio Processing Server

Local web server that serves the demo player and handles file uploads.
Supports single WAV file upload or batch upload of multiple files.
Processes each uploaded file through the HRTF binaural engine at
a user-selected position and returns the binaural WAV for playback.

Usage:
    cd claudio && .venv/bin/python -m claudio.audio_demo_server
    # Opens http://localhost:8787
"""
from __future__ import annotations

import io
import json
import os
import wave
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np

from claudio.audio_demo import process_through_claudio

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demo_output")
UPLOAD_DIR = os.path.join(DEMO_DIR, "uploads")
PORT = 8787


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Read WAV bytes into mono float32 array + sample rate."""
    with wave.open(io.BytesIO(data), "r") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit: unpack manually
        raw_bytes = bytearray(raw)
        n_samples = len(raw_bytes) // 3
        samples = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            b = raw_bytes[i * 3: i * 3 + 3]
            val = int.from_bytes(b, byteorder="little", signed=True)
            samples[i] = val / 8388608.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Downmix to mono if stereo
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples.astype(np.float32), sr


def _wav_bytes_stereo(left: np.ndarray, right: np.ndarray, sr: int) -> bytes:
    """Convert stereo arrays to WAV bytes."""
    buf = io.BytesIO()
    n = min(len(left), len(right))
    left_16 = np.clip(left[:n] * 32767, -32768, 32767).astype(np.int16)
    right_16 = np.clip(right[:n] * 32767, -32768, 32767).astype(np.int16)
    interleaved = np.empty(n * 2, dtype=np.int16)
    interleaved[0::2] = left_16
    interleaved[1::2] = right_16
    with wave.open(buf, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())
    return buf.getvalue()


POSITIONS = {
    "center":     [0.0, 0.0, -2.0],
    "left_30":    [-1.0, 0.0, -1.73],
    "left_45":    [-1.4, 0.0, -1.4],
    "left_90":    [-2.0, 0.0, 0.0],
    "right_30":   [1.0, 0.0, -1.73],
    "right_45":   [1.4, 0.0, -1.4],
    "right_90":   [2.0, 0.0, 0.0],
    "behind":     [0.0, 0.0, 2.0],
    "above":      [0.0, 2.0, -1.0],
}


class ClaudioHandler(SimpleHTTPRequestHandler):
    """HTTP handler with file upload and processing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DEMO_DIR, **kwargs)

    def do_POST(self):
        if self.path == "/api/process":
            self._handle_process()
        elif self.path == "/api/positions":
            self._send_json(POSITIONS)
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/api/positions":
            self._send_json(POSITIONS)
        elif self.path == "/upload" or self.path == "/upload/":
            self._serve_upload_page()
        else:
            super().do_GET()

    def _handle_process(self):
        content_length = int(self.headers.get("Content-Length", 0))
        content_type = self.headers.get("Content-Type", "")

        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        # Parse multipart manually
        boundary = content_type.split("boundary=")[1].strip()
        body = self.rfile.read(content_length)
        parts = self._parse_multipart(body, boundary)

        position_key = parts.get("position", "center")
        if position_key not in POSITIONS:
            position_key = "center"

        files_data = parts.get("_files", {})
        results = []

        for filename, file_bytes in files_data.items():
            if not filename.lower().endswith(".wav"):
                results.append({"filename": filename, "error": "Not a WAV file"})
                continue
            try:
                mono, sr = _read_wav(file_bytes)
                pos = np.array(POSITIONS[position_key])
                out_l, out_r = process_through_claudio(mono, pos, label=filename)
                wav_out = _wav_bytes_stereo(out_l, out_r, sr)

                # Save to uploads dir
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                out_name = f"{os.path.splitext(filename)[0]}_{position_key}_binaural.wav"
                out_path = os.path.join(UPLOAD_DIR, out_name)
                with open(out_path, "wb") as f:
                    f.write(wav_out)

                # Also save the original
                orig_path = os.path.join(UPLOAD_DIR, filename)
                with open(orig_path, "wb") as f:
                    f.write(file_bytes)

                results.append({
                    "filename": filename,
                    "original_url": f"/uploads/{filename}",
                    "binaural_url": f"/uploads/{out_name}",
                    "position": position_key,
                    "duration_s": len(mono) / sr,
                    "sample_rate": sr,
                })
            except Exception as e:
                results.append({"filename": filename, "error": str(e)})

        self._send_json({"results": results})

    def _parse_multipart(self, body: bytes, boundary: str) -> dict:
        """Minimal multipart/form-data parser."""
        parts = {}
        files = {}
        boundary_bytes = f"--{boundary}".encode()
        segments = body.split(boundary_bytes)

        for seg in segments[1:]:
            if seg.strip() == b"--" or seg.strip() == b"":
                continue
            header_end = seg.find(b"\r\n\r\n")
            if header_end < 0:
                continue
            header = seg[:header_end].decode("utf-8", errors="replace")
            content = seg[header_end + 4:]
            if content.endswith(b"\r\n"):
                content = content[:-2]

            # Extract name and filename
            name = ""
            filename = ""
            for line in header.split("\r\n"):
                if "name=" in line:
                    name_start = line.find('name="') + 6
                    name_end = line.find('"', name_start)
                    name = line[name_start:name_end]
                if "filename=" in line:
                    fn_start = line.find('filename="') + 10
                    fn_end = line.find('"', fn_start)
                    filename = line[fn_start:fn_end]

            if filename:
                files[filename] = content
            elif name:
                parts[name] = content.decode("utf-8", errors="replace")

        parts["_files"] = files
        return parts

    def _send_json(self, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_upload_page(self):
        html = _upload_page_html()
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"  [{self.log_date_time_string()}] {fmt % args}")


def _upload_page_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claudio — Upload & Process</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700;900&display=swap');
  :root {
    --bg: #0a0a0f; --card: #12121a; --border: #1e1e2e;
    --accent: #6366f1; --accent2: #a855f7;
    --dry: #f59e0b; --wet: #22d3ee;
    --text: #e2e8f0; --muted: #64748b;
    --success: #22c55e; --error: #ef4444;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 2rem;
  }
  .hero { text-align: center; padding: 2rem 1rem; }
  .hero h1 {
    font-size: 2.5rem; font-weight: 900;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { color: var(--muted); margin-top: 0.5rem; }
  .nav { text-align: center; margin: 1rem 0; }
  .nav a {
    color: var(--accent); text-decoration: none; font-weight: 500;
    padding: 0.5rem 1rem; border: 1px solid var(--accent);
    border-radius: 8px; transition: all 0.2s;
  }
  .nav a:hover { background: var(--accent); color: #fff; }
  .container { max-width: 800px; margin: 0 auto; }
  .upload-zone {
    border: 2px dashed var(--border); border-radius: 20px;
    padding: 3rem 2rem; text-align: center;
    transition: all 0.3s; cursor: pointer;
    background: var(--card); margin: 2rem 0;
  }
  .upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent);
    background: rgba(99,102,241,0.08);
    box-shadow: 0 0 40px rgba(99,102,241,0.1);
  }
  .upload-zone .icon { font-size: 3rem; margin-bottom: 1rem; }
  .upload-zone h3 { font-weight: 700; margin-bottom: 0.5rem; }
  .upload-zone p { color: var(--muted); font-size: 0.9rem; }
  .upload-zone input { display: none; }
  .controls {
    display: flex; gap: 1rem; align-items: center;
    justify-content: center; flex-wrap: wrap; margin: 1.5rem 0;
  }
  .controls label { color: var(--muted); font-size: 0.85rem; font-weight: 600; }
  .controls select {
    background: var(--card); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 0.6rem 1rem; font-family: inherit; font-size: 0.9rem;
    cursor: pointer;
  }
  .btn {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff; border: none; border-radius: 10px;
    padding: 0.8rem 2rem; font-weight: 700; font-size: 1rem;
    cursor: pointer; transition: all 0.2s; font-family: inherit;
  }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99,102,241,0.4); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  .file-list { margin: 1rem 0; }
  .file-item {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.6rem 1rem; background: var(--card);
    border: 1px solid var(--border); border-radius: 10px;
    margin-bottom: 0.5rem; font-size: 0.9rem;
  }
  .file-item .name { flex: 1; }
  .file-item .size { color: var(--muted); font-size: 0.8rem; }
  .file-item .remove {
    color: var(--error); cursor: pointer; font-weight: 700;
    width: 24px; height: 24px; display: flex; align-items: center;
    justify-content: center; border-radius: 50%;
    transition: background 0.2s;
  }
  .file-item .remove:hover { background: rgba(239,68,68,0.15); }
  .progress {
    background: var(--border); border-radius: 8px;
    height: 6px; margin: 1rem 0; overflow: hidden; display: none;
  }
  .progress .bar {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    height: 100%; border-radius: 8px; width: 0%;
    transition: width 0.3s;
  }
  .results { margin: 2rem 0; }
  .result-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    animation: fadeIn 0.4s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  .result-card h4 { font-weight: 600; margin-bottom: 0.8rem; }
  .result-card .ab-row { display: flex; gap: 1rem; flex-wrap: wrap; }
  .result-card .ab-col { flex: 1; min-width: 200px; }
  .result-card .label {
    display: inline-block; font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    padding: 0.2rem 0.5rem; border-radius: 5px; margin-bottom: 0.4rem;
  }
  .result-card .label.dry { background: rgba(245,158,11,0.15); color: var(--dry); }
  .result-card .label.wet { background: rgba(34,211,238,0.15); color: var(--wet); }
  .result-card audio { width: 100%; height: 36px; border-radius: 6px; }
  .result-card .meta { color: var(--muted); font-size: 0.8rem; margin-top: 0.5rem; }
  .result-card .error-msg { color: var(--error); font-size: 0.85rem; }
  .batch-info {
    text-align: center; color: var(--muted); font-size: 0.85rem;
    margin: 1rem 0; display: none;
  }
</style>
</head>
<body>
<div class="hero">
  <h1>Upload & Process</h1>
  <p>Drop your audio files, choose a position, and hear Claudio's binaural rendering</p>
</div>
<div class="nav">
  <a href="/">← Back to Demo Player</a>
</div>
<div class="container">
  <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <div class="icon">📁</div>
    <h3>Drop WAV files here</h3>
    <p>or click to browse — supports single files, multiple files, or entire folders</p>
    <input type="file" id="fileInput" accept=".wav,audio/wav" multiple webkitdirectory />
  </div>
  <div class="file-list" id="fileList"></div>
  <div class="controls">
    <label>Position:</label>
    <select id="posSelect">
      <option value="center">Center (0°)</option>
      <option value="left_30">Left 30°</option>
      <option value="left_45" selected>Left 45°</option>
      <option value="left_90">Hard Left (90°)</option>
      <option value="right_30">Right 30°</option>
      <option value="right_45">Right 45°</option>
      <option value="right_90">Hard Right (90°)</option>
      <option value="behind">Behind (180°)</option>
      <option value="above">Above</option>
    </select>
    <button class="btn" id="processBtn" disabled onclick="processFiles()">
      Process with Claudio
    </button>
  </div>
  <div class="batch-info" id="batchInfo"></div>
  <div class="progress" id="progress"><div class="bar" id="progressBar"></div></div>
  <div class="results" id="results"></div>
</div>
<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const processBtn = document.getElementById('processBtn');
const progress = document.getElementById('progress');
const progressBar = document.getElementById('progressBar');
const results = document.getElementById('results');
const batchInfo = document.getElementById('batchInfo');
let selectedFiles = [];

// Drag and drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  const items = e.dataTransfer.items;
  if (items) {
    const promises = [];
    for (const item of items) {
      const entry = item.webkitGetAsEntry && item.webkitGetAsEntry();
      if (entry) { promises.push(traverseEntry(entry)); }
      else if (item.kind === 'file') { selectedFiles.push(item.getAsFile()); }
    }
    Promise.all(promises).then(renderFileList);
  }
});

function traverseEntry(entry) {
  return new Promise(resolve => {
    if (entry.isFile) {
      entry.file(f => { if (f.name.toLowerCase().endsWith('.wav')) selectedFiles.push(f); resolve(); });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      reader.readEntries(entries => {
        Promise.all(entries.map(traverseEntry)).then(resolve);
      });
    } else { resolve(); }
  });
}

fileInput.addEventListener('change', () => {
  for (const f of fileInput.files) {
    if (f.name.toLowerCase().endsWith('.wav')) selectedFiles.push(f);
  }
  renderFileList();
});

function renderFileList() {
  fileList.innerHTML = selectedFiles.map((f, i) => `
    <div class="file-item">
      <span>🎵</span>
      <span class="name">${f.name}</span>
      <span class="size">${(f.size / 1024).toFixed(0)} KB</span>
      <span class="remove" onclick="removeFile(${i})">✕</span>
    </div>
  `).join('');
  processBtn.disabled = selectedFiles.length === 0;
  if (selectedFiles.length > 1) {
    batchInfo.style.display = 'block';
    batchInfo.textContent = selectedFiles.length + ' files selected — will process all at selected position';
  } else { batchInfo.style.display = 'none'; }
}

function removeFile(idx) {
  selectedFiles.splice(idx, 1);
  renderFileList();
}

async function processFiles() {
  if (selectedFiles.length === 0) return;
  const pos = document.getElementById('posSelect').value;
  processBtn.disabled = true;
  progress.style.display = 'block';
  progressBar.style.width = '10%';
  results.innerHTML = '';

  const total = selectedFiles.length;
  for (let i = 0; i < total; i++) {
    progressBar.style.width = ((i / total) * 80 + 10) + '%';
    const fd = new FormData();
    fd.append('position', pos);
    fd.append('file', selectedFiles[i]);
    try {
      const resp = await fetch('/api/process', { method: 'POST', body: fd });
      const data = await resp.json();
      for (const r of data.results) {
        if (r.error) {
          results.innerHTML += `<div class="result-card"><h4>${r.filename}</h4><p class="error-msg">❌ ${r.error}</p></div>`;
        } else {
          results.innerHTML += `
            <div class="result-card">
              <h4>${r.filename} — ${r.position.replace(/_/g, ' ')}</h4>
              <div class="meta">${r.duration_s.toFixed(1)}s · ${r.sample_rate} Hz</div>
              <div class="ab-row" style="margin-top:0.8rem">
                <div class="ab-col">
                  <span class="label dry">Original</span>
                  <audio controls src="${r.original_url}"></audio>
                </div>
                <div class="ab-col">
                  <span class="label wet">Binaural (Claudio)</span>
                  <audio controls src="${r.binaural_url}"></audio>
                </div>
              </div>
            </div>`;
        }
      }
    } catch (err) {
      results.innerHTML += `<div class="result-card"><h4>${selectedFiles[i].name}</h4><p class="error-msg">❌ Network error: ${err.message}</p></div>`;
    }
  }
  progressBar.style.width = '100%';
  setTimeout(() => { progress.style.display = 'none'; progressBar.style.width = '0%'; }, 800);
  processBtn.disabled = false;
}
</script>
</body>
</html>"""


def main():
    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Claudio Audio Processing Server                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  🎧 Demo player:  http://localhost:{PORT}/")
    print(f"  📁 Upload page:  http://localhost:{PORT}/upload")
    print(f"  📂 Serving from: {os.path.abspath(DEMO_DIR)}")
    print()

    server = HTTPServer(("", PORT), ClaudioHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
