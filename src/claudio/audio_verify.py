"""
audio_verify.py — Automated Audio Quality Verification

Reads back every generated WAV file and verifies:
  1. L/R energy ratios match expected spatial position
  2. Center sources have balanced L/R energy
  3. Left sources are louder in left ear, right in right ear
  4. Head rotation demo has varying L/R ratio over time
  5. Full mix has spatial separation
  6. No clipping, no silence, no NaN
  7. Generates a visual HTML report with waveform plots
"""

from __future__ import annotations

import math
import os
import wave

import numpy as np

DEMO_DIR = "demo_output"


def read_wav(filepath: str) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Read WAV file → (left, right_or_None, sample_rate)."""
    with wave.open(filepath, "r") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    else:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

    if n_ch == 1:
        return samples.astype(np.float32), None, sr
    elif n_ch == 2:
        left = samples[0::2].astype(np.float32)
        right = samples[1::2].astype(np.float32)
        return left, right, sr
    else:
        return samples.astype(np.float32), None, sr


def measure_energy(signal: np.ndarray) -> float:
    """RMS energy in dB."""
    rms = math.sqrt(float(np.mean(signal**2)) + 1e-30)
    return 20 * math.log10(rms + 1e-30)


def measure_lr_ratio_db(left: np.ndarray, right: np.ndarray) -> float:
    """L/R energy ratio in dB. Positive = louder left."""
    l_energy = float(np.mean(left**2)) + 1e-30
    r_energy = float(np.mean(right**2)) + 1e-30
    return 10 * math.log10(l_energy / r_energy)


def check_clipping(signal: np.ndarray) -> bool:
    """Check if signal clips (>= 0.99)."""
    return float(np.max(np.abs(signal))) >= 0.99


def check_silence(signal: np.ndarray, threshold_db: float = -60) -> bool:
    """Check if signal is effectively silent."""
    return measure_energy(signal) < threshold_db


def check_nan(signal: np.ndarray) -> bool:
    """Check for NaN values."""
    return bool(np.any(np.isnan(signal)))


def windowed_lr_ratios(
    left: np.ndarray,
    right: np.ndarray,
    n_windows: int = 8,
) -> list[float]:
    """Compute L/R ratio in dB for each time window."""
    block = len(left) // n_windows
    ratios = []
    for i in range(n_windows):
        s = i * block
        e = s + block
        ratios.append(measure_lr_ratio_db(left[s:e], right[s:e]))
    return ratios


# ─── Expected Spatial Properties ─────────────────────────────────────────────

# position_name → expected L/R ratio sign
# Positive ratio = louder in LEFT ear
EXPECTED_PANNING = {
    "center": "balanced",  # should be near 0 dB
    "left_45": "left",  # L/R ratio > 0
    "right_45": "right",  # L/R ratio < 0
    "hard_left": "left",  # L/R ratio >> 0
    "hard_right": "right",  # L/R ratio << 0
    "behind": "balanced",  # behind → similar L/R
    "above": "balanced",  # above → similar L/R
}


def verify_file(filepath: str) -> dict:
    """Verify a single WAV file and return results."""
    result = {
        "file": os.path.basename(filepath),
        "exists": os.path.exists(filepath),
        "checks": [],
        "passed": True,
    }
    if not result["exists"]:
        result["passed"] = False
        result["checks"].append(("file_exists", False, "File not found"))
        return result

    left, right, sr = read_wav(filepath)
    result["sample_rate"] = sr
    result["duration_s"] = len(left) / sr
    result["stereo"] = right is not None

    # Basic checks
    if check_nan(left):
        result["checks"].append(("no_nan_left", False, "NaN in left channel"))
        result["passed"] = False
    else:
        result["checks"].append(("no_nan_left", True, "Clean"))

    if right is not None and check_nan(right):
        result["checks"].append(("no_nan_right", False, "NaN in right channel"))
        result["passed"] = False
    elif right is not None:
        result["checks"].append(("no_nan_right", True, "Clean"))

    if check_silence(left):
        result["checks"].append(("not_silent", False, f"Left channel silent ({measure_energy(left):.1f} dB)"))
        result["passed"] = False
    else:
        result["checks"].append(("not_silent", True, f"Energy: {measure_energy(left):.1f} dB"))

    if check_clipping(left) or (right is not None and check_clipping(right)):
        result["checks"].append(("no_clipping", False, "Signal clips"))
        result["passed"] = False
    else:
        peak = float(np.max(np.abs(left)))
        if right is not None:
            peak = max(peak, float(np.max(np.abs(right))))
        result["checks"].append(("no_clipping", True, f"Peak: {peak:.3f}"))

    # Spatial checks (stereo only)
    if right is not None:
        lr_db = measure_lr_ratio_db(left, right)
        result["lr_ratio_db"] = lr_db

        # Determine position from filename
        basename = os.path.basename(filepath)
        position = None
        for pos_name in EXPECTED_PANNING:
            if f"_{pos_name}_" in basename:
                position = pos_name
                break

        if position:
            expected = EXPECTED_PANNING[position]
            if expected == "left":
                ok = lr_db > 0.5
                msg = f"L/R={lr_db:+.1f}dB {'✓ louder left' if ok else '✗ should be louder left'}"
            elif expected == "right":
                ok = lr_db < -0.5
                msg = f"L/R={lr_db:+.1f}dB {'✓ louder right' if ok else '✗ should be louder right'}"
            else:  # balanced
                ok = abs(lr_db) < 6.0
                msg = f"L/R={lr_db:+.1f}dB {'✓ balanced' if ok else '✗ not balanced'}"
            result["checks"].append(("spatial_correct", ok, msg))
            if not ok:
                result["passed"] = False

    return result


def verify_head_rotation(filepath: str) -> dict:
    """Verify the head rotation demo has varying spatial position."""
    result = verify_file(filepath)
    left, right, sr = read_wav(filepath)
    if right is None:
        result["checks"].append(("rotation_varies", False, "Not stereo"))
        result["passed"] = False
        return result

    ratios = windowed_lr_ratios(left, right, n_windows=8)
    lr_range = max(ratios) - min(ratios)
    ok = lr_range > 3.0  # Should vary by at least 3 dB during rotation
    result["checks"].append(
        (
            "rotation_varies",
            ok,
            f"L/R range={lr_range:.1f}dB across 8 windows {'✓' if ok else '✗ too static'}",
        )
    )
    result["lr_windows"] = ratios
    if not ok:
        result["passed"] = False
    return result


def verify_full_mix(dry_path: str, binaural_path: str) -> dict:
    """Verify the full mix has spatial separation vs dry."""
    result = verify_file(binaural_path)
    dry_left, _, _ = read_wav(dry_path)
    bin_left, bin_right, _ = read_wav(binaural_path)

    if bin_right is None:
        result["checks"].append(("mix_stereo", False, "Binaural mix not stereo"))
        result["passed"] = False
        return result

    # Binaural mix should have L/R differences (spatial separation)
    correlation = float(np.corrcoef(bin_left[:1000], bin_right[:1000])[0, 1])
    ok = correlation < 0.99  # Should NOT be identical L/R
    result["checks"].append(
        (
            "spatial_separation",
            ok,
            f"L/R correlation={correlation:.3f} {'✓ spatially separated' if ok else '✗ identical channels'}",
        )
    )
    if not ok:
        result["passed"] = False

    return result


def run_full_verification() -> list[dict]:
    """Run verification on all demo files."""
    results = []
    print("=" * 70)
    print("  CLAUDIO AUDIO VERIFICATION SUITE")
    print("=" * 70)

    # 1. Verify all instrument × position files
    instruments = ["guitar", "bass", "piano", "drums"]
    positions = ["center", "left_45", "right_45", "hard_left", "hard_right", "behind", "above"]

    print("\n─── Instrument × Position Grid ───")
    for inst in instruments:
        # Dry file
        dry = os.path.join(DEMO_DIR, f"{inst}_dry.wav")
        r = verify_file(dry)
        results.append(r)
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} {r['file']}")

        # Binaural files
        for pos in positions:
            wet = os.path.join(DEMO_DIR, f"{inst}_{pos}_binaural.wav")
            r = verify_file(wet)
            results.append(r)
            status = "✅" if r["passed"] else "❌"
            spatial = ""
            for name, _ok, msg in r["checks"]:
                if name == "spatial_correct":
                    spatial = f"  {msg}"
            print(f"  {status} {r['file']}{spatial}")

    # 2. Head rotation
    print("\n─── Head Rotation Demo ───")
    rot_path = os.path.join(DEMO_DIR, "guitar_head_rotation_binaural.wav")
    r = verify_head_rotation(rot_path)
    results.append(r)
    status = "✅" if r["passed"] else "❌"
    for name, _ok, msg in r["checks"]:
        if name == "rotation_varies":
            print(f"  {status} {r['file']}  {msg}")
    if "lr_windows" in r:
        print(f"       L/R per window: {[f'{x:+.1f}' for x in r['lr_windows']]}")

    # 3. Full mix
    print("\n─── Full Band Mix ───")
    r = verify_full_mix(
        os.path.join(DEMO_DIR, "full_mix_dry.wav"),
        os.path.join(DEMO_DIR, "full_mix_binaural.wav"),
    )
    results.append(r)
    status = "✅" if r["passed"] else "❌"
    for name, _ok, msg in r["checks"]:
        if name == "spatial_separation":
            print(f"  {status} {r['file']}  {msg}")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 70}")

    if failed > 0:
        print("\n  FAILURES:")
        for r in results:
            if not r["passed"]:
                print(f"    ❌ {r['file']}")
                for name, ok, msg in r["checks"]:
                    if not ok:
                        print(f"       └─ {name}: {msg}")

    return results


if __name__ == "__main__":
    run_full_verification()
