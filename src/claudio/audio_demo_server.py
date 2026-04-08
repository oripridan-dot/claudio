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
    from claudio.demo_upload_template import upload_page_html
    return upload_page_html()



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
