#!/usr/bin/env python3
"""
export_onnx.py — Export ForgeModel to ONNX for Claudio Studio AudioWorklet.

Usage:
  python export_onnx.py --checkpoint ./checkpoints/best.pt --output ./dist/synth.onnx
"""

from __future__ import annotations

import argparse

import torch

from claudio.forge.model.forge_model import ForgeModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     default="./dist/synth.onnx")
    p.add_argument("--clip-len",   type=int,   default=44100 * 3)
    args = p.parse_args()

    model = ForgeModel()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, args.clip_len)

    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy,
        args.output,
        input_names=["audio_in"],
        output_names=["audio_out"],
        dynamic_axes={"audio_in": {0: "batch", 1: "time"}, "audio_out": {0: "batch", 1: "time"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"✅ Exported → {args.output}")


if __name__ == "__main__":
    main()
