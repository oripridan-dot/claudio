import argparse
import os
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import torch
from model import DDSPDecoder
from synth import DDSPSynth


def export_model(output_path, checkpoint=None):
    model = DDSPDecoder()
    synth_state = None

    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading weights from {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"])
            if "synth_state_dict" in checkpoint_data:
                synth_state = checkpoint_data["synth_state_dict"]
        else:
            model.load_state_dict(checkpoint_data)
    else:
        print("No checkpoint found or specified, exporting with untrained weights.")

    model.eval()

    # Create dummy inputs matching the IntentFrame dimensions used in DDSPDecoder.ts
    # [batch, time, channels]
    dummy_f0 = torch.randn(1, 1, 1)
    dummy_loud = torch.randn(1, 1, 1)
    dummy_z = torch.randn(1, 1, 64)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_f0, dummy_loud, dummy_z),
        output_path,
        export_params=True,
        input_names=["f0", "loudness", "z"],
        output_names=["harmonics", "noise", "reverb_mix", "f0_residual", "voiced_mask"],
        dynamic_axes={
            "f0": {0: "batch", 1: "time"},
            "loudness": {0: "batch", 1: "time"},
            "z": {0: "batch", 1: "time"},
            "harmonics": {0: "batch", 1: "time"},
            "noise": {0: "batch", 1: "time"},
            "reverb_mix": {0: "batch", 1: "time"},
            "f0_residual": {0: "batch", 1: "time"},
            "voiced_mask": {0: "batch", 1: "time"},
        },
        opset_version=17,
    )
    print(f"✅ Neural Vocoder ONNX Exported successfully to: {output_path}")

    # Export the Learned Impulse Response (IR) for frontend Web Audio API
    if synth_state:
        # Load the synth to get access to its decay envelope mathematically
        synth = DDSPSynth(sample_rate=48000, frame_rate=250)
        synth.load_state_dict(synth_state)

        # ir_applied is the actual IR the convolution uses
        ir_applied = synth.reverb.ir * synth.reverb.decay_envelope
        ir_np = ir_applied.detach().cpu().numpy()

        # Normalize and convert to int16
        max_val = np.max(np.abs(ir_np))
        if max_val > 0:
            ir_np = ir_np / max_val
        ir_int16 = (ir_np * 32767).clip(-32768, 32767).astype("int16")

        ir_path = out_dir / "reverb_ir.wav"
        wav.write(str(ir_path), 48000, ir_int16)
        print(f"✅ Reverb IR Exported successfully directly to: {ir_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best.pt")
    # Default to placing it directly where the React frontend expects it!
    parser.add_argument("--output", type=str, default="../frontend/public/models/ddsp_model.onnx")
    args = parser.parse_args()

    export_model(args.output, args.checkpoint)
