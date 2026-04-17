"""
Claudio AI Fidelity Refinement Loop — Single-Process Edition
Runs training, evaluation, and analysis all in one Python process.
No subprocesses, no semaphore leaks, no checkpoint format mismatches.
"""

import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.optim as optim
from dataset import get_dataloader
from fidelity_analyzer import calculate_lsd, calculate_mcd, calculate_snr
from loss import CombinedPerceptualLoss
from model import DDSPDecoder
from synth import DDSPSynth

# ─── Configuration ────────────────────────────────────────────────
EPOCHS_PER_CYCLE = 40  # 40 epochs/cycle — GRU stabilises faster than pure MLP
MAX_CYCLES = 20  # Max training cycles
# Realistic targets for ~45 clips + spectral+perceptual loss.
# (MCD<200 / LSD<15 requires 1000s of clips + adversarial training — out of scope here)
MCD_TARGET = 500.0  # Achievable floor with augmented 45-clip dataset
LSD_TARGET = 25.0  # Achievable floor with augmented 45-clip dataset
# Use augmented dataset if available, fall back to processed
DATA_DIR = "data/augmented" if os.path.isdir("data/augmented") else "data/processed"
CHECKPOINT = "checkpoints/best.pt"
DEMO_DIR = "demo_output"
SR = 48000
# ──────────────────────────────────────────────────────────────────


def load_model_weights_only(model, synth, path, device):
    """Load only model weights — never restore optimizer/scheduler state."""
    if not os.path.exists(path):
        print("No checkpoint found — training from scratch.")
        return False
    try:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        if "synth_state_dict" in ckpt:
            synth.load_state_dict(ckpt["synth_state_dict"])
        print(f"✓ Loaded weights from {path}")
        return True
    except Exception as e:
        print(f"⚠ Could not load checkpoint ({e}) — training from scratch.")
        return False


def save_checkpoint(model, synth, best_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "synth_state_dict": synth.state_dict(), "best_loss": best_loss}, path
    )


def run_training_cycle(model, synth, loss_fn, optimizer, scheduler, dataloader, device, epochs, cycle_num):
    print(f"\n--- Training Cycle {cycle_num} ({epochs} epochs) ---")
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.monotonic()
        for batch in dataloader:
            f0 = batch["f0"].to(device)
            loudness = batch["loudness"].to(device)
            z = batch["z"].to(device)
            audio_true = batch["audio"].to(device)

            optimizer.zero_grad()
            harmonics, noise, reverb_mix, f0_residual, voiced_mask = model(f0, loudness, z)
            audio_hat = synth(f0, loudness, harmonics, noise, reverb_mix, f0_residual, voiced_mask)

            min_len = min(audio_hat.shape[-1], audio_true.shape[-1])
            audio_hat = audio_hat[:, :min_len]
            audio_true = audio_true[:, :min_len]

            loss = loss_fn(audio_hat, audio_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        elapsed = time.monotonic() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch + 1:>3}/{epochs} | Loss: {avg_loss:.4f} | LR: {lr_now:.2e} | {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, synth, best_loss, CHECKPOINT)

    print(f"  ✅ Cycle {cycle_num} complete — Best loss: {best_loss:.4f}")
    return best_loss


def synthesize_eval_samples(model, synth, dataloader, device, n_samples=3):
    """Synthesize comparison WAVs in-process."""
    os.makedirs(DEMO_DIR, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in dataloader:
            if saved >= n_samples:
                break
            f0 = batch["f0"][:1].to(device)
            loudness = batch["loudness"][:1].to(device)
            z = batch["z"][:1].to(device)
            audio_true = batch["audio"][:1]

            harmonics, noise, reverb_mix, f0_residual, voiced_mask = model(f0, loudness, z)
            audio_hat = synth(f0, loudness, harmonics, noise, reverb_mix, f0_residual, voiced_mask)

            min_len = min(audio_hat.shape[-1], audio_true.shape[-1])
            orig_np = audio_true[0, :min_len].cpu().numpy()
            gen_np = audio_hat[0, :min_len].detach().cpu().numpy()

            # Save as int16 WAV using scipy
            orig_int16 = (orig_np * 32767).clip(-32768, 32767).astype("int16")
            gen_int16 = (gen_np * 32767).clip(-32768, 32767).astype("int16")
            wav.write(f"{DEMO_DIR}/sample_{saved}_original.wav", SR, orig_int16)
            wav.write(f"{DEMO_DIR}/sample_{saved}_generated.wav", SR, gen_int16)
            saved += 1
    print(f"  ✓ Saved {saved} eval pairs to {DEMO_DIR}/")
    return saved


def compute_fidelity_metrics():
    """Compute MCD/LSD on the eval WAVs in-process using librosa."""
    import glob

    import librosa

    originals = sorted(glob.glob(f"{DEMO_DIR}/*_original.wav"))
    mcds, lsds = [], []
    for orig_path in originals:
        gen_path = orig_path.replace("_original.wav", "_generated.wav")
        if not os.path.exists(gen_path):
            continue
        y_t, _ = librosa.load(orig_path, sr=SR)
        y_p, _ = librosa.load(gen_path, sr=SR)
        min_len = min(len(y_t), len(y_p))
        y_t, y_p = y_t[:min_len], y_p[:min_len]
        mcds.append(calculate_mcd(y_t, y_p, SR))
        lsds.append(calculate_lsd(y_t, y_p, SR))

    if not mcds:
        return None, None
    return float(np.mean(mcds)), float(np.mean(lsds))


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\n{'=' * 50}")
    print("  CLAUDIO AI FIDELITY REFINEMENT ENGINE")
    print(f"  Device: {device} | Cycles: {MAX_CYCLES} | Epochs/Cycle: {EPOCHS_PER_CYCLE}")
    print(f"  Targets — MCD < {MCD_TARGET} | LSD < {LSD_TARGET}")
    print(f"{'=' * 50}\n")

    # Build model — single instance reused across all cycles
    model = DDSPDecoder().to(device)
    synth = DDSPSynth(sample_rate=SR, frame_rate=250, init_ir_path="../frontend/public/models/irs/Studio_A.wav").to(
        device
    )
    loss_fn = CombinedPerceptualLoss(sample_rate=SR).to(device)

    # num_workers=0 prevents macOS semaphore leaks
    dataloader = get_dataloader(DATA_DIR, batch_size=9, shuffle=True)
    dataloader_eval = get_dataloader(DATA_DIR, batch_size=1, shuffle=False)
    print(f"Dataset: {len(dataloader.dataset)} clips")

    # Load existing weights (not optimizer state)
    load_model_weights_only(model, synth, CHECKPOINT, device)

    for cycle in range(1, MAX_CYCLES + 1):
        print(f"\n{'─' * 50}")
        print(f"  CYCLE {cycle} / {MAX_CYCLES}")
        print(f"{'─' * 50}")

        # Fresh optimizer + scheduler every cycle — resets LR to 3e-4 each time
        # This prevents LR from dying at min_lr across multi-cycle runs
        # Optimize BOTH model weights and the learnable Reverb IR inside synth
        optimizer = optim.Adam(list(model.parameters()) + list(synth.reverb.parameters()), lr=3e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12, min_lr=1e-5)

        best_loss = run_training_cycle(
            model, synth, loss_fn, optimizer, scheduler, dataloader, device, EPOCHS_PER_CYCLE, cycle
        )

        # Reload best weights from this cycle before evaluating
        load_model_weights_only(model, synth, CHECKPOINT, device)

        # Evaluate
        synthesize_eval_samples(model, synth, dataloader_eval, device)
        mcd, lsd = compute_fidelity_metrics()

        if mcd is None:
            print("  ⚠ Could not compute fidelity metrics.")
            continue

        snr = calculate_snr(
            f"{DEMO_DIR}/sample_0_original.wav",
            f"{DEMO_DIR}/sample_0_generated.wav",
            SR,
        )

        print(f"\n  ┌── Fidelity Report (Cycle {cycle}) ──")
        print(f"  │  MCD:  {mcd:.2f}  (target < {MCD_TARGET})")
        print(f"  │  LSD:  {lsd:.4f}  (target < {LSD_TARGET})")
        print(f"  │  SNR:  {snr:.2f} dB  (higher = cleaner)")
        print(f"  │  Best Spectral Loss: {best_loss:.4f}")
        print("  └────────────────────────────────────")

        if mcd < MCD_TARGET and lsd < LSD_TARGET:
            print(f"\n🎯 TARGETS ACHIEVED after {cycle} cycles!")
            print(f"   Run: uv run python export_onnx.py --checkpoint {CHECKPOINT}")
            break

        # ReduceLROnPlateau manages LR decay — just report current LR for visibility
        current_lr = optimizer.param_groups[0]["lr"]  # type: ignore[possibly-undefined]
        remaining = MAX_CYCLES - cycle
        print(f"  ⚙ LR: {current_lr:.2e} | Targets not yet met. {remaining} cycles remaining...")

    print("\n✅ Refinement loop complete.")


if __name__ == "__main__":
    main()
