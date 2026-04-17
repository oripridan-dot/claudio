import argparse
import os
import time
import warnings

import torch
import torch.optim as optim
from dataset import get_dataloader
from loss import MultiScaleSpectralLoss
from model import DDSPDecoder
from synth import DDSPSynth


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Standard offline training setup — always start optimizer fresh
    model = DDSPDecoder().to(device)
    synth = DDSPSynth(sample_rate=48000, frame_rate=250).to(device)
    loss_fn = MultiScaleSpectralLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12, min_lr=1e-5)

    if os.path.exists("checkpoints/best.pt"):
        try:
            checkpoint = torch.load("checkpoints/best.pt", map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                # IMPORTANT: DO NOT restore optimizer/scheduler state — they may be frozen
                print("Loaded model weights from checkpoint. Starting with fresh optimizer LR=3e-4.")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded legacy weights. Starting with fresh optimizer LR=3e-4.")
        except BaseException as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")

    try:
        dataloader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    except FileNotFoundError:
        print(f"Dataset folder '{args.data_dir}' not found. Please run extract_dataset.py first.")
        return

    if len(dataloader.dataset) == 0:
        print(f"No .pt files found in '{args.data_dir}'. Provide data to train.")
        return

    print(f"Initializing Isolated Forge Training... {len(dataloader.dataset)} clips found.")

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.monotonic()

        for batch in dataloader:
            f0 = batch["f0"].to(device)
            loudness = batch["loudness"].to(device)
            z = batch["z"].to(device)
            audio_true = batch["audio"].to(device)

            # Forward pass Model -> (Harmonics, Noise params, Reverb Mix, F0 Residual, Voiced Mask)
            optimizer.zero_grad()
            harmonics, noise, reverb_mix, f0_residual, voiced_mask = model(f0, loudness, z)

            # Differentiable Synthesis -> Audio waveform
            audio_hat = synth(f0, loudness, harmonics, noise, reverb_mix, f0_residual, voiced_mask)

            # Pad or truncate to match exactly due to upsampling rounding
            min_len = min(audio_hat.shape[-1], audio_true.shape[-1])
            audio_hat = audio_hat[:, :min_len]
            audio_true = audio_true[:, :min_len]

            # Loss computation
            loss = loss_fn(audio_hat, audio_true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        # Real-time epoch-level refinement trap:
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{args.epochs} - Spectral Loss: {avg_loss:.4f} - Time: {time.monotonic() - t0:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                "checkpoints/best.pt",
            )

    print(f"✅ Training complete. Best loss: {best_loss:.4f} saved to checkpoints/best.pt")


if __name__ == "__main__":
    main()
