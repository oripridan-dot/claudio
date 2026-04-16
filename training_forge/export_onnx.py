import argparse
import os
from pathlib import Path

import torch
from model import DDSPDecoder


def export_model(output_path, checkpoint=None):
    model = DDSPDecoder()

    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading weights from {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
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
        input_names=['f0', 'loudness', 'z'],
        output_names=['harmonics', 'noise'],
        dynamic_axes={
            'f0': {0: 'batch', 1: 'time'},
            'loudness': {0: 'batch', 1: 'time'},
            'z': {0: 'batch', 1: 'time'},
            'harmonics': {0: 'batch', 1: 'time'},
            'noise': {0: 'batch', 1: 'time'},
        },
        opset_version=17
    )
    print(f"✅ Neural Vocoder ONNX Exported successfully to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best.pt")
    # Default to placing it directly where the React frontend expects it!
    parser.add_argument("--output", type=str, default="../frontend/public/models/ddsp_model.onnx")
    args = parser.parse_args()

    export_model(args.output, args.checkpoint)
