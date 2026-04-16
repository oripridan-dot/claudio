import os

import soundfile as sf
import torch
from dataset import get_dataloader
from model import DDSPDecoder
from synth import DDSPSynth


def evaluate(checkpoint_path="checkpoints/best.pt", data_dir="data/processed", output_dir="demo_output"):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading test data...")
    dataloader = get_dataloader(data_dir, batch_size=1, shuffle=True)

    decoder = DDSPDecoder()
    if os.path.exists(checkpoint_path):
        decoder.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    decoder.eval()
    synth = DDSPSynth(sample_rate=48000, frame_rate=250)
    synth.eval()

    print("Evaluating and synthesizing samples...")
    # Take first 3 clips for evaluation
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break

        f0 = batch['f0']
        loudness = batch['loudness']
        z = batch['z']
        orig_audio = batch['audio']

        with torch.no_grad():
            harmonics, noise = decoder(f0, loudness, z)
            gen_audio = synth(f0, loudness, harmonics, noise)

        gen_audio = gen_audio.squeeze().cpu().numpy()
        orig_audio = orig_audio.squeeze().cpu().numpy()

        orig_path = os.path.join(output_dir, f"sample_{i}_original.wav")
        gen_path = os.path.join(output_dir, f"sample_{i}_generated.wav")

        sf.write(orig_path, orig_audio, 48000)
        sf.write(gen_path, gen_audio, 48000)
        print(f"Saved: {orig_path} and {gen_path}")

    print("Evaluation complete. Fidelity check ready in demo_output/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="demo_output")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_dir, args.output_dir)
