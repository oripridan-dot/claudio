#!/usr/bin/env python3
"""
train.py — Claudio Analog Forge training entry-point.

Usage:
  python train.py --data ./data/violin --epochs 100 --batch-size 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from claudio.forge.data.audio_dataset import AudioDataset
from claudio.forge.loss.spectral_loss import MultiScaleSpectralLoss
from claudio.forge.model.forge_model import ForgeModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Claudio Analog Forge trainer")
    p.add_argument("--data",        default="./data",  help="Path to WAV dataset directory")
    p.add_argument("--epochs",  type=int, default=100, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--checkpoint", default="./checkpoints", help="Output directory for checkpoints")
    p.add_argument("--clip-seconds", type=float, default=3.0, help="Audio clip length")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = (
        torch.device("cuda")  if args.device == "auto" and torch.cuda.is_available() else
        torch.device("mps")   if args.device == "auto" and torch.backends.mps.is_available() else
        torch.device(args.device) if args.device != "auto" else
        torch.device("cpu")
    )
    print(f"🔧 Device: {device}")

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = AudioDataset(args.data, clip_seconds=args.clip_seconds)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"📂 Dataset: {len(dataset)} clips from {args.data}")

    model     = ForgeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = MultiScaleSpectralLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.monotonic()

        for audio in loader:
            audio = audio.to(device)
            optimizer.zero_grad()
            audio_hat = model(audio)
            loss = criterion(audio_hat, audio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        elapsed = time.monotonic() - t0
        print(f"Epoch {epoch:04d}/{args.epochs} | loss={avg:.4f} | {elapsed:.1f}s")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  ✓ checkpoint saved (loss={best_loss:.4f})")

    print(f"\n✅ Training complete. Best loss: {best_loss:.4f}")
    print(f"   Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
