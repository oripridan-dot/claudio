import os
import numpy as np
import scipy.signal as signal
import soundfile as sf

SR = 48000
DURATION = 7  # seconds for testing stems
OUT_DIR = "data/calibration_stems"


def save_wav(name: str, y: np.ndarray):
    """Normalize and save waveform to .wav"""
    peak = np.max(np.abs(y)) + 1e-9
    # Always normalize extreme signals
    y = y / peak
    sf.write(os.path.join(OUT_DIR, f"{name}.wav"), y, SR)


def generate_white_noise(length: int):
    return np.random.normal(0, 1, length)


def generate_brown_noise(length: int):
    white = generate_white_noise(length)
    brown = np.cumsum(white)
    # Remove DC offset
    return brown - np.mean(brown)


def generate_blue_noise(length: int):
    white = generate_white_noise(length + 1)
    blue = np.diff(white)
    return blue


def generate_pink_noise(length: int):
    # Quick 1/f generation using FFT filtering
    white = generate_white_noise(length)
    X = np.fft.rfft(white)
    # create 1/f filter
    f = np.fft.rfftfreq(length)
    f[0] = 1e-9  # avoid div by zero
    X = X / np.sqrt(f)
    pink = np.fft.irfft(X, n=length)
    return pink


def generate_black_noise(length: int):
    # Absolute silence for threshold testing
    return np.zeros(length)


def generate_fm_sweep(length: int):
    t = np.linspace(0, DURATION, length)
    # Sweep from 20Hz to 16000Hz logarithmically to test tracking delta latency
    return signal.chirp(t, f0=20, f1=16000, t1=DURATION, method="logarithmic")


def generate_impulse_train(length: int):
    y = np.zeros(length)
    # Dirac Spike every 500ms to test transient smearing
    hop = SR // 2
    y[::hop] = 1.0
    return y


def generate_pitch_staircase(length: int):
    y = np.zeros(length)
    # Hard-stepping frequencies to test f0 residual snapping
    t = np.arange(length) / SR
    freqs = [110, 220, 330, 440, 550, 660, 880, 1760]
    for i in range(length):
        sec = int(t[i])
        f_idx = min(sec, len(freqs) - 1)
        y[i] = np.sin(2 * np.pi * freqs[f_idx] * t[i])
    return y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    length = int(SR * DURATION)

    print("🤖 Booting Claudio Robot Calibration Camp...")

    noises = {
        "calib_white_noise": generate_white_noise,
        "calib_pink_noise": generate_pink_noise,
        "calib_brown_noise": generate_brown_noise,
        "calib_blue_noise": generate_blue_noise,
        "calib_black_noise": generate_black_noise,
        "calib_logic_fm_sweep": generate_fm_sweep,
        "calib_logic_impulses": generate_impulse_train,
        "calib_logic_staircase": generate_pitch_staircase,
    }

    for name, func in noises.items():
        print(f"  -> Generating {name}...")
        y = func(length)
        save_wav(name, y)

    print(f"\n✅ Calibration Camp complete. High-fidelity stress stems written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
