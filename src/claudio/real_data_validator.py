import os
import time
import json
import wave
import numpy as np
import logging
from typing import Dict, List, Any

# Internal imports
from claudio.intent.intent_encoder import IntentEncoder
from claudio.intent.intent_decoder import IntentDecoder
from claudio.intent.intent_protocol import IntentPacket
from claudio.hrtf_engine import HRTFBinauralEngine, AudioSource
from claudio.signal_flow_config import SignalFlowConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Claudio.RealDataValidator")

FIXTURE_DIR = "/Users/oripridan/ANTIGRAVITY/claudio/tests/audio_fixtures"
OUTPUT_DIR = "demo_output/validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, 'rb') as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        data = wf.readframes(n_frames)
        # Assuming 16-bit mono for simplicity, convert to float32 [-1, 1]
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        # If stereo, just take left channel for intent extraction
        if wf.getnchannels() == 2:
            samples = samples[0::2]
        return samples, sr

def write_wav(path: str, data: np.ndarray, sr: int):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1 if data.ndim == 1 else 2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # Convert [-1, 1] to int16
        raw = (data * 32767.0).astype(np.int16).tobytes()
        wf.writeframes(raw)

class RealDataValidator:
    def __init__(self):
        self.report = {
            "timestamp": time.time(),
            "tiers": {
                "intent_fidelity": {"passed": False, "tests": []},
                "spatial_arena": {"passed": False, "tests": []},
                "multi_peer": {"passed": False, "tests": []}
            }
        }

    def validate_intent_fidelity(self):
        logger.info("--- TIER A: Intent Fidelity Validation ---")
        fixtures = ["saxophone.wav", "piano.wav", "drum_kit.wav"]
        
        for name in fixtures:
            path = os.path.join(FIXTURE_DIR, name)
            if not os.path.exists(path):
                logger.warning(f"Fixture {name} not found, skipping.")
                continue
                
            samples, sr = read_wav(path)
            encoder = IntentEncoder(sample_rate=sr)
            decoder = IntentDecoder() # Falls back to additive if model missing
            
            # Process in larger blocks to satisfy frame_len requirements
            intents = []
            reconstructed_audio = []
            
            # IntentEncoder needs at least hop*8 samples per frame
            # At 44.1k: 176*8 = 1408 samples. 2048 is safe.
            block_size = 2048 
            for i in range(0, len(samples) - block_size, block_size):
                chunk = samples[i:i+block_size]
                frames = encoder.encode_block(chunk, start_time_ms=(i/sr)*1000.0)
                if frames:
                    intents.extend(frames)
                    # Decode back to audio
                    decoded_chunk = decoder.decode_frames(frames)
                    reconstructed_audio.append(decoded_chunk)
            
            if not reconstructed_audio:
                logger.error(f"Failed to extract any intents from {name}. Audio too short or window mismatch.")
                self.report["tiers"]["intent_fidelity"]["tests"].append({
                    "fixture": name,
                    "passed": False,
                    "error": "No frames extracted"
                })
                continue

            reconstructed = np.concatenate(reconstructed_audio)
            out_path = os.path.join(OUTPUT_DIR, f"recon_{name}")
            write_wav(out_path, reconstructed, sr)
            
            # Basic metrics
            f0_errs = [abs(f.f0_hz - 0) for f in intents if f.f0_hz > 0] # Simple check if tracked anything
            passed = len(f0_errs) > 0
            
            self.report["tiers"]["intent_fidelity"]["tests"].append({
                "fixture": name,
                "frames_processed": len(intents),
                "passed": passed,
                "reconstructed_file": out_path
            })
            logger.info(f"Fixture {name}: Captured {len(intents)} intent frames. Passed: {passed}")

    def validate_spatial_arena(self):
        logger.info("--- TIER C: Spatial Arena Verification ---")
        cfg = SignalFlowConfig()
        engine = HRTFBinauralEngine(config=cfg)
        
        # Test 1: Center position
        src_center = AudioSource(source_id="center", position=np.array([0.0, 0.0, -2.0]))
        engine.add_source(src_center)
        
        # Feed 1 second of white noise
        sr = cfg.render_sample_rate
        noise = (np.random.randn(sr) * 0.3).astype(np.float32)
        
        out_l_all, out_r_all = [], []
        block = cfg.fft_size
        for i in range(0, len(noise) - block, block):
            chunk = noise[i:i+block]
            frame = engine.render({"center": chunk})
            out_l_all.append(frame.left)
            out_r_all.append(frame.right)
            
        l_final = np.concatenate(out_l_all)
        r_final = np.concatenate(out_r_all)
        
        l_energy = np.mean(l_final**2)
        r_energy = np.mean(r_final**2)
        ratio_db = 10 * np.log10(l_energy / r_energy)
        
        # Center should be balanced within 0.5dB
        passed = float(abs(ratio_db)) < 0.5
        self.report["tiers"]["spatial_arena"]["tests"].append({
            "test": "center_balance",
            "lr_ratio_db": float(round(ratio_db, 3)),
            "passed": passed
        })
        logger.info(f"Center balance test: {ratio_db:+.3f} dB. Passed: {passed}")

    def generate_report(self):
        # Overall pass check
        for tier in self.report["tiers"].values():
            tier["passed"] = all(t["passed"] for t in tier["tests"]) if tier["tests"] else False
            
        report_path = os.path.join(OUTPUT_DIR, "validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2)
            
        # Markdown Report
        md_path = "validation_report.md"
        with open(md_path, 'w') as f:
            f.write("# Claudio Full Validation Report\n\n")
            f.write(f"Generated at: {time.ctime(self.report['timestamp'])}\n\n")
            
            for name, tier in self.report["tiers"].items():
                status = "✅ PASSED" if tier["passed"] else "❌ FAILED"
                f.write(f"## Tier: {name.replace('_', ' ').title()} - {status}\n")
                if not tier["tests"]:
                    f.write("No tests run.\n\n")
                    continue
                f.write("| Test | Result | Info |\n")
                f.write("| --- | --- | --- |\n")
                for t in tier["tests"]:
                    t_status = "✅" if t["passed"] else "❌"
                    info = t.get("reconstructed_file", t.get("lr_ratio_db", ""))
                    f.write(f"| {t.get('fixture', t.get('test', '???'))} | {t_status} | {info} |\n")
                f.write("\n")
        
        logger.info(f"Reports generated: {report_path}, {md_path}")

if __name__ == "__main__":
    validator = RealDataValidator()
    validator.validate_intent_fidelity()
    validator.validate_spatial_arena()
    # Tier B involves multi-process server interaction, will handle separately or in next check
    validator.generate_report()
