"""
train_forge.py — Polyphonic AutoEncoder Training Script

Trains the End-to-End Convolutional Latent Audio Codec on raw audio.
This bypasses monophonic limitations and allows synthesis of drums & chords.

Usage:
    python scripts/train_forge.py [--epochs 300] [--lr 3e-4] [--batch-size 4]

Device: MPS (Apple Silicon) → CUDA → CPU (auto-detect)
Output: checkpoints/forge_model_best.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from claudio.forge.loss.spectral_loss import MultiScaleSpectralLoss  # noqa: E402
from claudio.forge.model.autoencoder import AudioAutoEncoder  # noqa: E402


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_wav(path: Path, sr: int = 44_100) -> np.ndarray:
    """Load a WAV file as mono float32 numpy array."""
    import wave

    with wave.open(str(path)) as wf:
        raw = wf.readframes(wf.getnframes())
        n_ch = wf.getnchannels()
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)
    return samples


def extract_raw_clips(
    data_dir: str,
    clip_seconds: float = 3.0,
    sample_rate: int = 44_100,
    n_clips_per_file: int = 20,
) -> torch.Tensor:
    """Extract random raw audio clips from all WAV files for self-supervised training."""
    wav_files = sorted(Path(data_dir).rglob("*.wav")) + sorted(Path(data_dir).rglob("*.WAV"))
    wav_files = [f for f in wav_files if not f.name.startswith("._")]
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {data_dir}")

    clip_len = int(clip_seconds * sample_rate)
    audio_list = []

    print(f"Extracting {clip_seconds}s clips from {len(wav_files)} files...")
    for path in wav_files:
        print(f"  Processing: {path.name}", end="", flush=True)
        t0 = time.time()

        audio = load_wav(path, sample_rate)

        rng = np.random.default_rng(42)
        for _ in range(n_clips_per_file):
            if len(audio) <= clip_len:
                clip = np.pad(audio, (0, clip_len - len(audio)))
            else:
                start = rng.integers(0, len(audio) - clip_len)
                clip = audio[start:start + clip_len]

            clip_t = torch.from_numpy(clip).unsqueeze(0)  # (1, T)
            audio_list.append(clip_t)

        elapsed = time.time() - t0
        print(f" ({elapsed:.1f}s)")

    audio_all = torch.cat(audio_list, dim=0)   # (N, T)
    print(f"  Total clips: {audio_all.shape[0]}, Shape: {audio_all.shape}")
    return audio_all


def train(
    data_dir: str = "data/calibration",
    epochs: int = 300,
    lr: float = 3e-4,
    batch_size: int = 8,
    clip_seconds: float = 3.0,
    checkpoint_dir: str = "checkpoints",
    n_clips_per_file: int = 20,
) -> dict:
    """Train ForgeModel synthesis path on pre-extracted features."""
    device = get_device()
    print(f"Device: {device}")
    print(f"Data:   {data_dir}")
    print(f"Config: epochs={epochs}, lr={lr}, batch_size={batch_size}")
    print()

    # ── Phase 1: Extract Audio Clips ──────────────────────────────────
    audio_all = extract_raw_clips(
        data_dir, clip_seconds, n_clips_per_file=n_clips_per_file,
    )

    dataset = TensorDataset(audio_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ── Phase 2: Build model (AudioAutoEncoder) ───────────────────────
    latent_dim = 128
    autoencoder = AudioAutoEncoder(latent_dim=latent_dim).to(device)

    n_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"\nSynthesis model (Polyphonic AutoEncoder): {n_params:,} parameters")

    # ── Loss + Optimizer ──────────────────────────────────────────────
    loss_fn_spectral = MultiScaleSpectralLoss().to(device)
    loss_fn_l1 = torch.nn.L1Loss().to(device)
    params = list(autoencoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )

    # ── Checkpoint setup ──────────────────────────────────────────────
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(exist_ok=True)
    best_loss = float("inf")
    best_path = ckpt_path / "forge_model_best.pt"

    # ── Training loop ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Training Polyphonic Neural Codec")
    print("=" * 60)

    t0 = time.time()
    losses_history: list[float] = []

    for epoch in range(1, epochs + 1):
        autoencoder.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            audio_batch = batch[0].to(device)  # (B, T)

            optimizer.zero_grad()

            # Forward: Raw Audio -> AutoEncoder -> Reconstructed Audio
            pred_audio = autoencoder(audio_batch)  # (B, T)

            # Combined Loss: Spectral fidelity + Transient Structure (L1)
            l_spectral = loss_fn_spectral(pred_audio, audio_batch)
            l_l1 = loss_fn_l1(pred_audio, audio_batch)
            loss = l_spectral + 10.0 * l_l1

            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses_history.append(avg_loss)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "autoencoder_state_dict": autoencoder.state_dict(),
                "loss": best_loss,
                "n_params": n_params,
                "latent_dim": latent_dim,
            }, best_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"Loss: {avg_loss:.6f} | "
                f"Best: {best_loss:.6f} | "
                f"LR: {lr_now:.2e} | "
                f"Time: {elapsed:.0f}s"
            )

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Epochs:       {epochs}")
    print(f"  Final loss:   {losses_history[-1]:.6f}")
    print(f"  Best loss:    {best_loss:.6f}")
    print(f"  Convergence:  {losses_history[0]:.6f} → {losses_history[-1]:.6f}")
    improvement = (1 - losses_history[-1] / max(losses_history[0], 1e-10)) * 100
    print(f"  Improvement:  {improvement:.1f}%")
    print(f"  Time:         {elapsed:.0f}s ({elapsed / epochs:.2f}s/epoch)")
    print(f"  Checkpoint:   {best_path}")

    return {
        "epochs": epochs,
        "final_loss": losses_history[-1],
        "best_loss": best_loss,
        "initial_loss": losses_history[0],
        "improvement_pct": improvement,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Claudio DDSP Synthesis")
    parser.add_argument("--data-dir", default="data/calibration")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--clips-per-file", type=int, default=20)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        clip_seconds=args.clip_seconds,
        checkpoint_dir=args.checkpoint_dir,
        n_clips_per_file=args.clips_per_file,
    )
