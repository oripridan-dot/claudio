"""
stress_tester.py — Concurrent Load Simulator for Claudio WebSocket Protocol

Simulates N clients connecting to M collab rooms simultaneously, streaming
binary intent packets at 120Hz and capturing JSON signaling.

Validates:
  - Uvicorn/FastAPI async throughput
  - DDSP neural batching bottlenecks (if enabled)
  - Broadcast multiplexing efficiency
"""

import asyncio
import argparse
import time
import struct
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import urllib.request
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

async def simulate_client(client_id: int, room_id: str, ddsp_enabled: bool, duration: int):
    uri = f"ws://127.0.0.1:8000/ws/collab/{room_id}?name=Simulated_{client_id}"
    
    try:
        async with websockets.connect(uri, ping_interval=None) as ws:
            # Enable DDSP if requested
            if ddsp_enabled:
                await ws.send(json.dumps({"type": "ddsp_toggle", "enabled": True}))
                
            start = time.time()
            packets_sent = 0
            packets_received = 0
            audio_bytes_received = 0
            seq = 0
            
            async def receive_loop():
                nonlocal packets_received, audio_bytes_received
                try:
                    async for msg in ws:
                        if isinstance(msg, bytes):
                            if msg.startswith(b"DDSP"):
                                audio_bytes_received += len(msg)
                            else:
                                packets_received += 1
                except websockets.ConnectionClosed:
                    pass
            
            rx_task = asyncio.create_task(receive_loop())
            
            # Send loop (120Hz = 8.3ms)
            while time.time() - start < duration:
                seq += 1
                # Dummy binary packet (FULL FRAME ~34 bytes padding)
                # Header: seq(I) ts(f) flags(B) = 9 bytes + dummy float payload
                packet = struct.pack("<IfB", seq, time.time() - start, 0x01) + (b'\\x00' * 25)
                await ws.send(packet)
                packets_sent += 1
                await asyncio.sleep(1/120)
                
            # Finish
            await ws.close()
            rx_task.cancel()
            
            return {
                "id": client_id,
                "sent": packets_sent,
                "rx": packets_received,
                "audio_bytes": audio_bytes_received
            }
            
    except Exception as e:
        return {"id": client_id, "error": str(e)}

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=10, help="Number of concurrent clients")
    parser.add_argument("--rooms", type=int, default=2, help="Number of rooms to distribute clients into")
    parser.add_argument("--duration", type=int, default=5, help="Test duration in seconds")
    parser.add_argument("--ddsp", action="store_true", help="Enable server-side DDSP inference")
    args = parser.parse_args()
    
    logging.info(f"🚀 Starting Stress Test: {args.clients} clients over {args.rooms} rooms for {args.duration}s. DDSP={'ON' if args.ddsp else 'OFF'}")
    
    # Pre-create rooms
    room_ids = []
    for _ in range(args.rooms):
        req = urllib.request.Request("http://127.0.0.1:8000/api/collab/create", method="POST")
        try:
            with urllib.request.urlopen(req) as res:
                data = json.loads(res.read())
                room_ids.append(data["room_id"])
        except Exception as e:
            logging.error(f"Failed to create room: {e}")
            return
            
    logging.info(f"Created rooms: {room_ids}")
    
    # Launch clients
    tasks = []
    for i in range(args.clients):
        room_id = room_ids[i % args.rooms]
        tasks.append(simulate_client(i, room_id, args.ddsp, args.duration))
        
    results = await asyncio.gather(*tasks)
    
    # Aggregate
    total_sent = 0
    total_rx = 0
    total_audio = 0
    errors = 0
    for res in results:
        if "error" in res:
            errors += 1
            logging.error(f"Client {res['id']} failed: {res['error']}")
        else:
            total_sent += res["sent"]
            total_rx += res["rx"]
            total_audio += res["audio_bytes"]
            
    logging.info("-" * 40)
    logging.info(f"🎯 Stress Test Complete")
    logging.info(f"  Packets Sent:     {total_sent}")
    logging.info(f"  Packets Received: {total_rx}")
    logging.info(f"  DDSP Audio Rx:    {total_audio / 1024:.1f} KB")
    logging.info(f"  Failed Clients:   {errors}")
    logging.info("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
