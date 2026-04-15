import asyncio
import websockets
import httpx
import json
import time
import os
import wave
import secrets
import numpy as np
import logging
from dataclasses import asdict

from claudio.intent.intent_encoder import IntentEncoder
from claudio.intent.intent_protocol import IntentStream, IntentPacket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Claudio.StressSim")

FIXTURE_DIR = "/Users/oripridan/ANTIGRAVITY/claudio/tests/audio_fixtures"
HTTP_BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"

def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, 'rb') as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        data = wf.readframes(n_frames)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() == 2:
            samples = samples[0::2]
        return samples, sr

class SimulatedMusician:
    def __init__(self, name: str, fixture_name: str, room_id: str):
        self.name = name
        self.fixture_name = fixture_name
        self.room_id = room_id
        self.samples = None
        self.sr = None
        self.packets_sent = 0
        self.packets_received = 0
        self.running = True

    async def get_token(self):
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{HTTP_BASE}/api/auth/token", json={"username": self.name})
            return res.json()["token"]

    async def connect(self):
        try:
            token = await self.get_token()
            url = f"{WS_BASE}/ws/collab/{self.room_id}?name={self.name}&role=both&token={token}"
            logger.info(f"[{self.name}] Connecting to {url}...")
            
            path = os.path.join(FIXTURE_DIR, self.fixture_name)
            self.samples, self.sr = read_wav(path)
            
            encoder = IntentEncoder(sample_rate=self.sr)
            stream = IntentStream()
            
            async with websockets.connect(url) as ws:
                logger.info(f"[{self.name}] Connected.")
                
                # Start listener
                listener_task = asyncio.create_task(self.listen(ws))
                
                # Stream intents
                hop = self.sr // 250
                block_size = hop * 8 # Match Encoder window
                
                start_time = time.time()
                frame_count = 0
                
                # We limit the simulation to 5 seconds or the end of the file
                limit_samples = min(len(self.samples), self.sr * 5)
                
                while frame_count * hop + block_size < limit_samples and self.running:
                    start_sample = frame_count * hop
                    chunk = self.samples[start_sample : start_sample + block_size]
                    
                    frames = encoder.encode_block(chunk, start_time_ms=(start_sample/self.sr)*1000.0)
                    if frames:
                        for frame in frames:
                            packet = stream.pack(frame)
                            data = packet.to_bytes()
                            await ws.send(data)
                            self.packets_sent += 1
                    
                    frame_count += 1
                    # Real-time pacing: 250 FPS -> 4ms per frame
                    await asyncio.sleep(0.004)
                
                logger.info(f"[{self.name}] Finished streaming {self.packets_sent} packets.")
                self.running = False
                await listener_task
                
        except Exception as e:
            logger.error(f"[{self.name}] Connection failed: {e}")

    async def listen(self, ws):
        try:
            while self.running:
                msg = await ws.recv()
                self.packets_received += 1
                # We don't need to parse everything, just count
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[{self.name}] Connection closed.")
        except Exception as e:
            logger.error(f"[{self.name}] Listener error: {e}")

async def run_stress_test():
    room_id = "autonomous-band-v6"
    async with httpx.AsyncClient() as client:
        # Create the fixed simulation room
        try:
            res = await client.post(f"{HTTP_BASE}/api/collab/create", json={"username": "SimulationOrchestrator"})
            if res.status_code == 200:
                data = res.json()
                # If the server supports requesting a specific ID, we'd use it. 
                # For now, we'll try to use the one returned or just use the fixed one if it's open.
                # Actually, let's just use the room_id the server gives us and tell the user!
                room_id = data["room_id"]
        except Exception as e:
            logger.warning(f"Could not create room via API, attempting to use fallback: {e}")

    musicians = [
        SimulatedMusician("AI-Sax", "saxophone.wav", room_id),
        SimulatedMusician("AI-Piano", "piano.wav", room_id),
        SimulatedMusician("AI-Drums", "drum_kit.wav", room_id),
        SimulatedMusician("AI-Bass", "bass_guitar.wav", room_id),
    ]
    
    logger.info(f"--- STARTING SUSTAINED SIMULATION IN ROOM: {room_id} ---")
    logger.info(f"--- PLEASE JOIN THIS ROOM IN THE UI TO OBSERVE ---")
    
    # Run for 30 minutes
    duration_seconds = 30 * 60
    start_sim = time.time()
    
    while time.time() - start_sim < duration_seconds:
        tasks = [m.connect() for m in musicians]
        await asyncio.gather(*tasks)
        if time.time() - start_sim < duration_seconds:
            logger.info("Simulation loop restarting...")
            await asyncio.sleep(2) # Brief pause before restart
            for m in musicians:
                m.running = True
                m.packets_sent = 0
                m.packets_received = 0
    
    # Summary
    logger.info("--- STRESS TEST SUMMARY ---")
    for m in musicians:
        logger.info(f"{m.name}: Sent {m.packets_sent}, Received {m.packets_received}")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
