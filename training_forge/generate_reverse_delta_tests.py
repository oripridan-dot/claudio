import os
import onnxruntime as ort
import numpy as np
import torch
import scipy.io.wavfile as wavfile
from synth import DDSPSynth

SAMPLE_RATE = 48000
FRAME_RATE = 250


def infer_and_render(f0, loudness, z, name, filename, onnx_sess, synth):
    print(f"\n--- Testing: {name} ---")
    # The frontend processes frames one by one. We should do the same to respect
    # the static graph operations (e.g., reshape to [1, 64]) if any exist.
    frames = f0.shape[1]

    harmonics_list, noise_list, reverb_mix_list, f0_residual_list, voiced_mask_list = [], [], [], [], []

    for t in range(frames):
        inputs = {"f0": f0[:, t : t + 1, :], "loudness": loudness[:, t : t + 1, :], "z": z[:, t : t + 1, :]}
        results = onnx_sess.run(None, inputs)

        output_names = [o.name for o in onnx_sess.get_outputs()]
        idx_map = {n: i for i, n in enumerate(output_names)}

        harmonics_list.append(results[idx_map.get("harmonics", 0)])
        noise_list.append(results[idx_map.get("noise", 1)])
        reverb_mix_list.append(results[idx_map.get("reverb_mix", 2)])
        f0_residual_list.append(results[idx_map.get("f0_residual", 3)])
        voiced_mask_list.append(results[idx_map.get("voiced_mask", 4)])

    harmonics = np.concatenate(harmonics_list, axis=1)
    noise = np.concatenate(noise_list, axis=1)
    reverb_mix = np.concatenate(reverb_mix_list, axis=1)
    f0_residual = np.concatenate(f0_residual_list, axis=1)
    voiced_mask = np.concatenate(voiced_mask_list, axis=1)

    # Reverse Delta Analysis
    if name == "Pure Sine Sweep":
        variance = np.var(f0_residual)
        mean_res = np.mean(np.abs(f0_residual))
        print(f"[{name}] f0_residual Variance (target: 0.0): {variance:.6f}")
        print(f"[{name}] f0_residual Mean Absolute Error: {mean_res:.6f}")
        if variance > 0.05:
            print("[WARNING] Neural network is aggressively pitch-bending a mathematically perfect sweep.")
        else:
            print("[PASS] Pitch correlation is stable.")

    # Render audio using the differentiable synth
    audio = synth(
        torch.from_numpy(f0).float(),
        torch.from_numpy(loudness).float(),
        torch.from_numpy(harmonics).float(),
        torch.from_numpy(noise).float(),
        torch.from_numpy(reverb_mix).float(),
        torch.from_numpy(f0_residual).float(),
        torch.from_numpy(voiced_mask).float(),
    )
    audio_np = audio.detach().cpu().numpy()[0]

    if name == "Absolute Void":
        max_val = np.max(np.abs(audio_np))
        db_level = 20 * np.log10(max_val + 1e-10)
        print(f"[{name}] Noise Floor True Peak: {db_level:.2f} dB")
        if db_level > -80.0:
            print("[WARNING] V/UV Gate is bleeding DC Offset or ambient rumble during absolute silence.")
        else:
            print("[PASS] Acoustic void maintained.")

    if name == "Dirac Impulse":
        peak_idx = np.argmax(np.abs(audio_np))
        post_peak = np.abs(audio_np[peak_idx:])

        # drops below -60dB of the peak
        threshold = np.max(post_peak) * 0.001
        below_thresh = np.where(post_peak < threshold)[0]
        tail_len = below_thresh[0] if len(below_thresh) > 0 else len(post_peak)
        tail_ms = (tail_len / SAMPLE_RATE) * 1000.0

        print(f"[{name}] Transient Decay Time (-60dB): {tail_ms:.2f} ms")
        if tail_ms > 20.0:
            print("[WARNING] Massive transient smearing detected. Gate is too sluggish.")
        else:
            print("[PASS] Razor sharp transient response.")

    # Save to disk
    max_amp = np.max(np.abs(audio_np))
    if max_amp > 1.0:
        audio_np = audio_np / max_amp
    wav_data = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    wavfile.write(filename, SAMPLE_RATE, wav_data)
    print(f"✅ Rendered output -> {filename}")


def main():
    model_path = "/Users/oripridan/ANTIGRAVITY/claudio/frontend/public/models/ddsp_model.onnx"
    output_dir = "/Users/oripridan/ANTIGRAVITY/claudio/training_forge/demo_output"
    if not os.path.exists(model_path):
        print(f"[FATAL] Model not found at {model_path}")
        return

    print("Initializing Reverse Delta Tests...")
    print(f"Loading ONNX Model: {model_path}")
    onnx_sess = ort.InferenceSession(model_path)
    synth = DDSPSynth(SAMPLE_RATE, FRAME_RATE)
    synth.eval()

    # 1. The Pure Sine Sweep
    # 5 seconds, f0 from 20 to 20k
    frames_sweep = int(5.0 * FRAME_RATE)
    f0_sweep = np.linspace(20, 20000, frames_sweep, dtype=np.float32).reshape(1, frames_sweep, 1)
    loudness_sweep = np.full((1, frames_sweep, 1), 0.8, dtype=np.float32)
    z_sweep = np.zeros((1, frames_sweep, 64), dtype=np.float32)
    infer_and_render(
        f0_sweep, loudness_sweep, z_sweep, "Pure Sine Sweep", f"{output_dir}/reverse_delta_sweep.wav", onnx_sess, synth
    )

    # 2. The Dirac Impulse
    # 1 second, single 4ms frame of max loudness
    frames_impulse = int(1.0 * FRAME_RATE)
    f0_impulse = np.zeros((1, frames_impulse, 1), dtype=np.float32)
    loudness_impulse = np.zeros((1, frames_impulse, 1), dtype=np.float32)
    loudness_impulse[0, frames_impulse // 2, 0] = 1.0
    z_impulse = np.zeros((1, frames_impulse, 64), dtype=np.float32)
    infer_and_render(
        f0_impulse,
        loudness_impulse,
        z_impulse,
        "Dirac Impulse",
        f"{output_dir}/reverse_delta_impulse.wav",
        onnx_sess,
        synth,
    )

    # 3. The Absolute Void
    # 5 seconds, pure silence
    frames_void = int(5.0 * FRAME_RATE)
    f0_void = np.zeros((1, frames_void, 1), dtype=np.float32)
    loudness_void = np.zeros((1, frames_void, 1), dtype=np.float32)
    z_void = np.zeros((1, frames_void, 64), dtype=np.float32)
    infer_and_render(
        f0_void, loudness_void, z_void, "Absolute Void", f"{output_dir}/reverse_delta_void.wav", onnx_sess, synth
    )


if __name__ == "__main__":
    main()
