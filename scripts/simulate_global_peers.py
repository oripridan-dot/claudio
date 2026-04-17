#!/usr/bin/env python3
import asyncio
import json
import sys
import urllib.request
import wave

import numpy as np
import websockets

from claudio.intent.intent_encoder import FRAME_RATE_HZ, IntentEncoder
from claudio.intent.intent_protocol import IntentStream

PEER_CONFIGS = [
    {"name": "Tokyo-Bass", "file": "data/calibration/AllHailThePowerOfJesusName_Bass_E.wav", "instrument": "Bass"},
    {"name": "London-Keys", "file": "data/calibration/AllHailThePowerOfJesusName_Piano_E.wav", "instrument": "Piano"},
    {"name": "NY-Vocal", "file": "data/calibration/AllHailThePower_VocalGuide.wav", "instrument": "Vocal"},
]


async def simulate_peer(ws_url: str, config: dict):
    print(f"[{config['name']}] Connecting to {ws_url}...")
    try:
        async with websockets.connect(ws_url) as ws:
            print(f"[{config['name']}] Connected! Sending instrument metadata...")
            await ws.send(json.dumps({"type": "instrument_set", "instrument": config["instrument"]}))

            encoder = IntentEncoder(sample_rate=44100)
            stream = IntentStream()

            with wave.open(config["file"], "rb") as wf:
                if wf.getnchannels() != 1:
                    print(f"[{config['name']}] Requires mono audio.")
                    return
                if wf.getframerate() != 44100:
                    print(f"[{config['name']}] Resampling not supported in simulator (must be 44.1kHz).")
                    return

                chunk_size = encoder.frame_len
                print(f"[{config['name']}] Streaming intent packets @ {FRAME_RATE_HZ}Hz...")

                start_ms = 0.0
                while True:
                    data = wf.readframes(chunk_size)
                    if len(data) < chunk_size * 2:
                        wf.rewind()  # Loop forever
                        continue

                    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    frames = encoder.encode_block(audio_chunk, start_time_ms=start_ms)

                    for frame in frames:
                        packet = stream.pack(frame)
                        await ws.send(packet.to_bytes())

                    start_ms += len(audio_chunk) / 44.1
                    # Emulate real-time processing
                    await asyncio.sleep(len(audio_chunk) / 44100.0)
    except Exception as e:
        print(f"[{config['name']}] Disconnected: {e}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python simulate_global_peers.py <server_url> <room_id>")
        print("Example: python simulate_global_peers.py ws://localhost:8000 b7a1c4")
        sys.exit(1)

    server_url = sys.argv[1].rstrip("/")
    room_id = sys.argv[2]

    http_url = server_url.replace("ws://", "http://").replace("wss://", "https://")
    req = urllib.request.Request(
        f"{http_url}/api/auth/token", data=b'{"username":"simulator"}', headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        token = json.loads(response.read())["token"]

    tasks = []
    for cfg in PEER_CONFIGS:
        ws_url = f"{server_url}/ws/collab/{room_id}?name={cfg['name']}&token={token}"
        tasks.append(simulate_peer(ws_url, cfg))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
