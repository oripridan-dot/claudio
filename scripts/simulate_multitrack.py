import os

import numpy as np
import scipy.signal
import soundfile as sf

from claudio.audio_demo import write_wav_stereo
from claudio.hrtf_engine import AudioSource, HRTFBinauralEngine
from claudio.intent.intent_decoder import IntentDecoder
from claudio.intent.intent_encoder import IntentEncoder
from claudio.signal_flow_config import SignalFlowConfig

MULTITRACK_DIR = "/Users/oripridan/Downloads/AllHailThePowerOfJesusName_MULTITRACKS_WAV"
OUTPUT_PATH = "/Users/oripridan/ANTIGRAVITY/claudio/demo_output/multitrack_claudio_mix.wav"


def main():
    sr = 44100
    cfg = SignalFlowConfig(capture_sample_rate=sr, render_sample_rate=sr, fft_size=512, hrir_length=256)
    engine = HRTFBinauralEngine(config=cfg)

    if not os.path.exists(MULTITRACK_DIR):
        print(f"Error: {MULTITRACK_DIR} not found.")
        return

    files = [f for f in os.listdir(MULTITRACK_DIR) if f.endswith(".wav") and not f.startswith("._")]
    files.sort()

    # We will process 15 seconds of audio for a good demo output.
    duration_s = 15.0
    max_samples = int(duration_s * sr)

    positions = [
        np.array([-2.0, 0.0, -1.0]),  # left
        np.array([2.0, 0.0, -1.0]),  # right
        np.array([-1.0, 0.5, -2.0]),  # center left
        np.array([1.0, 0.5, -2.0]),  # center right
        np.array([0.0, -0.5, -3.0]),  # bottom
        np.array([0.0, 1.0, -2.0]),  # top
        np.array([-3.0, 0.0, 0.0]),  # hard left
        np.array([3.0, 0.0, 0.0]),  # hard right
    ]

    decoder_model = "/Users/oripridan/ANTIGRAVITY/claudio/checkpoints/forge_model_best.pt"

    processed_tracks = {}
    max_len = 0

    print(f"Loading and converting up to 8 tracks from {MULTITRACK_DIR}...")
    for i, file in enumerate(files[:8]):
        filepath = os.path.join(MULTITRACK_DIR, file)
        try:
            waveform, file_sr = sf.read(filepath)
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            continue

        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)

        if file_sr != sr:
            num_samples = int(len(waveform) * float(sr) / file_sr)
            waveform = scipy.signal.resample(waveform, num_samples)

        waveform = waveform.astype(np.float32)

        # Trim / Pad to exact duration
        actual_len = len(waveform)
        if actual_len > max_samples:
            waveform = waveform[:max_samples]
            actual_len = max_samples
        elif actual_len < max_samples:
            padded = np.zeros(max_samples, dtype=np.float32)
            padded[:actual_len] = waveform
            waveform = padded
            actual_len = max_samples

        if actual_len > max_len:
            max_len = actual_len

        print(f"[{i + 1}/8] Translating {file} to Intent Packets -> DDSP Neural Decoder...")

        encoder = IntentEncoder(sample_rate=sr)
        decoder = IntentDecoder(sample_rate=sr, model_path=decoder_model)

        chunk_size = sr
        n_chunks = max_samples // chunk_size
        decoded_chunks = []
        for c in range(n_chunks):
            chunk = waveform[c * chunk_size : (c + 1) * chunk_size]

            # 1. Semantic pipeline (Mentors/UI)
            frames = encoder.encode_block(chunk, start_time_ms=c * 1000.0)

            # 2. Audio Latent pipeline (Polyphonic transport)
            if getattr(decoder, "use_ddsp", False):
                d_chunk = decoder.decode_raw_audio(chunk)
            else:
                d_chunk = decoder.decode_frames(frames)

            # Add artificial jitter / packet drop to simulate global network (1% drop)
            if np.random.rand() < 0.01:
                # Dropped frame packet!
                d_chunk = np.zeros_like(d_chunk)

            decoded_chunks.append(d_chunk)

        if decoded_chunks:
            regen_audio = np.concatenate(decoded_chunks)
            pos = positions[i % len(positions)]
            processed_tracks[file] = {"audio": regen_audio, "pos": pos}
            src = AudioSource(source_id=file, position=pos)
            engine.add_source(src)
            print(f"      Regenerated layer mapped to spatial node {pos}.")

    print("\nRendering Global Binaural Mix (Claudio Engine)...")
    block = cfg.fft_size
    n_blocks = max_len // block
    out_l_parts, out_r_parts = [], []

    for b in range(n_blocks):
        buffers = {}
        for name, data in processed_tracks.items():
            chunk = data["audio"][b * block : (b + 1) * block]
            if len(chunk) == block:
                buffers[name] = chunk
            else:
                pad = np.zeros(block, dtype=np.float32)
                pad[: len(chunk)] = chunk
                buffers[name] = pad

        frame = engine.render(buffers)
        out_l_parts.append(frame.left)
        out_r_parts.append(frame.right)

    out_l = np.concatenate(out_l_parts)
    out_r = np.concatenate(out_r_parts)

    peak = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-6)
    out_l *= 0.85 / peak
    out_r *= 0.85 / peak

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    write_wav_stereo(OUTPUT_PATH, out_l, out_r, sr)
    print(f"\n✅ Binaural Spatial Mix saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
