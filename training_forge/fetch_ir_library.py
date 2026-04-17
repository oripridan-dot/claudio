import os
import numpy as np
import soundfile as sf
import scipy.signal as signal

SR = 48000
FRONTEND_IR_DIR = "frontend/public/models/irs"


def generate_algorithmic_ir(name: str, decay_time: float, early_reflections: int, high_cut: float):
    """
    Generates a high-quality physical algorithmic reverberation impulse response
    using decaying noise filtered through a low-pass filter to simulate high-frequency absorption.
    """
    length = int(SR * decay_time)

    # 1. Base exponentially decaying noise
    t = np.linspace(0, decay_time, length)
    noise = np.random.normal(0, 1, length)

    # Define decay envelope: E(t) = e^(-6.91 * t / T) (T60 definition)
    envelope = np.exp(-6.91 * t / decay_time)
    ir = noise * envelope

    # 2. Add early reflections (sparse Diracs to simulate wall bounces)
    for _ in range(early_reflections):
        idx = np.random.randint(50, 2000)  # between ~1ms and 40ms
        if idx < length:
            ir[idx] += np.random.uniform(0.5, 1.0)

    # 3. Simulate high-frequency air dampening (low-pass filter)
    nyquist = SR / 2
    b, a = signal.butter(1, high_cut / nyquist, btype="low")
    ir_filtered = signal.lfilter(b, a, ir)

    # Normalize
    peak = np.max(np.abs(ir_filtered)) + 1e-9
    ir_filtered = ir_filtered / peak

    return ir_filtered


def main():
    os.makedirs(FRONTEND_IR_DIR, exist_ok=True)

    print("🏛️ Compiling the Claudio IR Library (Algorithmically via Physical Modeling)...")

    irs = {
        "Studio_A.wav": {"decay": 0.4, "reflections": 3, "cut": 12000},
        "Cathedral.wav": {"decay": 3.5, "reflections": 15, "cut": 4000},
        "EMT_Plate.wav": {"decay": 1.2, "reflections": 30, "cut": 8000},
        "Claudio_Ambient.wav": {"decay": 2.0, "reflections": 5, "cut": 6000},
    }

    for filename, params in irs.items():
        print(f"  -> Baking IR geometry: {filename}...")
        y = generate_algorithmic_ir(filename, params["decay"], params["reflections"], params["cut"])
        dest = os.path.join(FRONTEND_IR_DIR, filename)
        sf.write(dest, y, SR)
        print(f"    [ok] Saved {filename}")

    print(f"\n✅ IR Library established at {FRONTEND_IR_DIR}/")


if __name__ == "__main__":
    main()
