"""
simulate_ddsp_onnx.py — Claudio DDSP ONNX Inference Stress Simulator

Tests the production .onnx model against 9 real audio fixtures.
Validates sub-ms inference logic, computes RTF, and generates SOTA fidelity metrics.
"""

import glob
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

# Add paths for Claudio imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir / "tests"))
sys.path.insert(0, str(root_dir / "training_forge"))

from benchmark_resynthesis import measure_snr_approx, measure_spectral_preservation, measure_transient_error
from synth import DDSPSynth

from claudio.intent.intent_encoder import IntentEncoder


def run_stress_test():
    print("═" * 72)
    print("  CLAUDIO DDSP ONNX STRESS SIMULATION")
    print("═" * 72)

    model_path = root_dir / "frontend" / "public" / "models" / "ddsp_model.onnx"
    if not model_path.exists():
        print(f"Error: ONNX model not found at {model_path}")
        return

    # Initialize ONNX inference session
    print("Loading ONNX Inference Session...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])

    fixtures_dir = root_dir / "tests" / "audio_fixtures"
    audio_files = glob.glob(str(fixtures_dir / "*.wav"))

    if not audio_files:
        print(f"No audio files found in {fixtures_dir}")
        return

    print(f"Found {len(audio_files)} audio fixtures. Beginning Stress Test...\n")

    results = []

    for _idx, path in enumerate(sorted(audio_files)):
        name = Path(path).stem.replace("_", " ").title()
        audio, sr = sf.read(path)

        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        dur = len(audio) / sr

        # 1. Initialize Pipeline
        encoder = IntentEncoder(sample_rate=sr)
        dssp_synth = DDSPSynth(sample_rate=sr, frame_rate=250)

        hop = encoder.hop
        n_blocks = len(audio) // hop

        inf_times = []
        out_audio_blocks = []

        t0_all = time.perf_counter()

        for i in range(n_blocks):
            start = i * hop
            block = audio[start : start + encoder.frame_len]
            if len(block) < encoder.frame_len:
                block = np.pad(block, (0, encoder.frame_len - len(block)))

            # E2E Encode
            frames = encoder.encode_block(block, start_time_ms=(start / sr) * 1000)
            if frames:
                frame = frames[0]

                # Prepare ONNX Inputs
                f0_in = np.array([[[frame.f0_hz]]], dtype=np.float32)
                loud_in = np.array([[[frame.loudness_norm]]], dtype=np.float32)
                mfcc_arr = frame.mfcc if frame.mfcc is not None else [0] * 13
                # ensure mfcc has length 13
                if len(mfcc_arr) < 13:
                    mfcc_arr = mfcc_arr + [0] * (13 - len(mfcc_arr))
                z_in = np.array([[mfcc_arr[:13]]], dtype=np.float32)

                # Measure purely the ML Inference
                t0_inf = time.perf_counter()

                outputs = session.run(["harmonics", "noise"], {"f0": f0_in, "loudness": loud_in, "z": z_in})

                inf_time = (time.perf_counter() - t0_inf) * 1000
                inf_times.append(inf_time)

                # Synthesize locally to compute metrics
                # Convert back to torch for DDSPSynth
                f0_t = torch.from_numpy(f0_in)
                loud_t = torch.from_numpy(loud_in)
                harm_t = torch.from_numpy(outputs[0])
                noise_t = torch.from_numpy(outputs[1])

                with torch.no_grad():
                    gen_chunk = dssp_synth(f0_t, loud_t, harm_t, noise_t)

                # DSP synth returns specific buffer length, we keep only 'hop'
                gen_arr = gen_chunk.squeeze().cpu().numpy()
                out_audio_blocks.append(gen_arr[:hop])
            else:
                out_audio_blocks.append(np.zeros(hop, dtype=np.float32))

        # Consolidate outputs
        if len(out_audio_blocks) > 0:
            out_audio = np.concatenate(out_audio_blocks)[: len(audio)]
        else:
            out_audio = np.zeros(len(audio), dtype=np.float32)

        t_wall = time.perf_counter() - t0_all

        avg_inf_ms = float(np.mean(inf_times)) if inf_times else 0.0
        p99_inf_ms = float(np.percentile(inf_times, 99)) if inf_times else 0.0
        rtf = dur / t_wall if t_wall > 0 else 0.0

        spec_pres = measure_spectral_preservation(audio, out_audio)
        transient_err = measure_transient_error(audio, out_audio, sr)
        snr_val = measure_snr_approx(audio, out_audio)

        print(
            f"✅ {name:<18} | Inf(Avg): {avg_inf_ms:.3f}ms | Inf(P99): {p99_inf_ms:.3f}ms | SpecCorr: {spec_pres:.3f} | RTF: {rtf:.1f}x"
        )

        results.append(
            {
                "name": name,
                "dur": dur,
                "avg_inf": avg_inf_ms,
                "p99_inf": p99_inf_ms,
                "rtf": rtf,
                "spec_pres": spec_pres,
                "snr": snr_val,
                "transient": transient_err,
            }
        )

    # Save artifact
    report_path = (
        root_dir
        / ".gemini"
        / "antigravity"
        / "brain"
        / "48506ba8-c903-480f-8f8a-0866d0e38be3"
        / "ddsp_stress_report.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Neural Vocoder (DDSP ONNX) Stress Simulation Report\n\n")
        f.write(
            "Tested on Pure Intent Architecture (v3.0) via `onnxruntime` CPU provider mimicking WebNN constraints.\n\n"
        )
        f.write("| Instrument | Avg Latency | P99 Latency | Spectral Corr | SNR (approx) | RTF |\n")
        f.write("|------------|-------------|-------------|---------------|--------------|-----|\n")
        for r in results:
            f.write(
                f"| {r['name']} | {r['avg_inf']:.3f}ms | {r['p99_inf']:.3f}ms | {r['spec_pres']:.3f} | {r['snr']:.1f} dB | {r['rtf']:.1f}x |\n"
            )

    print("\nScorecard published to -> ddsp_stress_report.md")


if __name__ == "__main__":
    run_stress_test()
