"""
hrtf_data.py — HRTF Synthesis and Lookup

Procedural HRTF synthesis using Woodworth-Schlosberg ITD model
and Brown-Duda head-shadow ILD approximation. Provides HRIR pairs
for arbitrary azimuth/elevation angles with configurable resolution.

This module is imported by hrtf_engine.py and handles all HRTF data
concerns independently of the rendering engine.
"""

from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 192_000
HRIR_LEN = 256
SPEED_OF_SOUND = 343.0

_HRTF_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


# ─── Quaternion Utilities ────────────────────────────────────────────────────


def azimuth_elevation_from_position(
    pos: np.ndarray,
    head_quat: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Convert a 3D source position to azimuth/elevation relative to head."""
    rotated = _quat_rotate_vector(pos, _quat_conjugate(head_quat))
    x, y, z = rotated
    dist = math.sqrt(x**2 + y**2 + z**2) + 1e-8
    # -Z is the listener's forward direction in Claudio's coordinate system
    azimuth = math.degrees(math.atan2(x, -z))
    elevation = math.degrees(math.asin(y / dist))
    return azimuth, elevation


def _quat_conjugate(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)


def _quat_rotate_vector(
    v: np.ndarray,
    q: tuple[float, float, float, float],
) -> np.ndarray:
    w, qx, qy, qz = q
    vx, vy, vz = v[0], v[1], v[2]
    t_x = 2 * (qy * vz - qz * vy)
    t_y = 2 * (qz * vx - qx * vz)
    t_z = 2 * (qx * vy - qy * vx)
    return np.array(
        [
            vx + w * t_x + qy * t_z - qz * t_y,
            vy + w * t_y + qz * t_x - qx * t_z,
            vz + w * t_z + qx * t_y - qy * t_x,
        ],
        dtype=np.float64,
    )


# ─── HRIR Synthesis ──────────────────────────────────────────────────────────


def get_hrir(
    azimuth_deg: float,
    elevation_deg: float,
    hrir_len: int = HRIR_LEN,
    sample_rate: int = SAMPLE_RATE,
    grid_res: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesise HRIR pair using Woodworth-Schlosberg ITD
    and frequency-dependent head-shadow ILD.
    """
    az_key = round(azimuth_deg / grid_res) * int(grid_res)
    el_key = round(elevation_deg / 10) * 10
    cache_key = (az_key % 360, el_key)

    if cache_key in _HRTF_CACHE:
        cached = _HRTF_CACHE[cache_key]
        if len(cached[0]) == hrir_len:
            return cached

    az_rad = math.radians(az_key)
    el_rad = math.radians(el_key)

    # Woodworth-Schlosberg ITD
    head_radius = 0.0875
    abs_az = abs(az_rad)
    if abs_az <= math.pi / 2:
        itd_s = (head_radius / SPEED_OF_SOUND) * (math.sin(abs_az) + abs_az)
    else:
        itd_s = (head_radius / SPEED_OF_SOUND) * (1.0 + math.pi / 2)
    itd_s *= math.cos(el_rad)
    itd_samples = int(itd_s * sample_rate)
    itd_samples = min(itd_samples, hrir_len // 4)

    # Frequency-dependent ILD (Brown-Duda head shadow)
    # Use absolute azimuth — the branching below assigns sides correctly
    ild_db = 8.0 * abs(math.sin(az_rad)) * math.cos(el_rad)

    hrir_l = np.zeros(hrir_len, dtype=np.float32)
    hrir_r = np.zeros(hrir_len, dtype=np.float32)
    ild_linear = 10 ** (ild_db / 20.0)  # always >= 1.0

    if az_rad >= 0:
        # Source on RIGHT: right ear is ipsilateral (louder, earlier)
        hrir_r[0] = ild_linear
        hrir_l[itd_samples] = 1.0 / ild_linear
    else:
        # Source on LEFT: left ear is ipsilateral (louder, earlier)
        hrir_l[0] = ild_linear
        hrir_r[itd_samples] = 1.0 / ild_linear

    # Early reflections for realistic spectral content
    # Reflections preserve spatial cue: ipsilateral ear gets more energy
    rng = np.random.default_rng(seed=abs(az_key * 1000 + el_key))
    tail_start = max(itd_samples + 1, 4)
    n_ref = min(8, (hrir_len - tail_start) // 4)
    # Spatial bias: reflections favor ipsilateral ear
    # Spatial bias: reflections favor ipsilateral ear
    abs_sin_az = abs(math.sin(az_rad))
    ipsi_bias = 0.5 + 0.3 * abs_sin_az  # 0.5 at center/behind, 0.8 at 90°
    for r_idx in range(n_ref):
        tap = tail_start + r_idx * (hrir_len // (n_ref + 1))
        if tap < hrir_len:
            decay = 0.08 * math.exp(-0.4 * r_idx)
            # Scale jitter by laterality — no jitter at center/behind
            jitter = rng.uniform(-0.3, 0.3) * abs_sin_az
            if az_rad >= 0:
                hrir_r[tap] += decay * (ipsi_bias + jitter)
                hrir_l[tap] += decay * (1 - ipsi_bias - jitter)
            else:
                hrir_l[tap] += decay * (ipsi_bias + jitter)
                hrir_r[tap] += decay * (1 - ipsi_bias - jitter)

    # Elevation pinna filter — continuous spectral notch
    abs_el = abs(elevation_deg)
    if abs_el > 5.0:
        notch_depth = min(0.8, abs_el / 90.0)
        notch_idx = int(hrir_len * max(0.1, 1.0 - abs_el / 90.0) * 0.4)
        notch_idx = max(2, min(notch_idx, hrir_len - 1))
        hrir_l[notch_idx] *= 1.0 - notch_depth
        hrir_r[notch_idx] *= 1.0 - notch_depth
        comb_delay = int(max(3, 12 - abs_el * 0.1))
        if comb_delay < hrir_len:
            hrir_l[comb_delay] += 0.1 * math.sin(math.radians(abs_el))
            hrir_r[comb_delay] += 0.1 * math.sin(math.radians(abs_el))

    # NOTE: Onset fade removed — it caused L/R asymmetry by attenuating
    # the leading impulse in the ipsilateral channel differently than the
    # delayed impulse in the contralateral channel.

    _HRTF_CACHE[cache_key] = (hrir_l, hrir_r)
    return hrir_l, hrir_r


def interpolate_hrir_bilinear(
    azimuth_deg: float,
    elevation_deg: float,
    hrir_len: int,
    sample_rate: int,
    grid_res: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear spherical interpolation between 4 adjacent grid points."""
    az_lo = math.floor(azimuth_deg / grid_res) * grid_res
    az_hi = az_lo + grid_res
    el_lo = math.floor(elevation_deg / 10) * 10
    el_hi = el_lo + 10

    frac_az = (azimuth_deg - az_lo) / grid_res if grid_res > 0 else 0
    frac_el = (elevation_deg - el_lo) / 10.0
    frac_az = max(0.0, min(1.0, frac_az))
    frac_el = max(0.0, min(1.0, frac_el))

    ll_l, ll_r = get_hrir(az_lo, el_lo, hrir_len, sample_rate, grid_res)
    hl_l, hl_r = get_hrir(az_hi, el_lo, hrir_len, sample_rate, grid_res)
    lh_l, lh_r = get_hrir(az_lo, el_hi, hrir_len, sample_rate, grid_res)
    hh_l, hh_r = get_hrir(az_hi, el_hi, hrir_len, sample_rate, grid_res)

    w00 = (1 - frac_az) * (1 - frac_el)
    w10 = frac_az * (1 - frac_el)
    w01 = (1 - frac_az) * frac_el
    w11 = frac_az * frac_el

    interp_l = w00 * ll_l + w10 * hl_l + w01 * lh_l + w11 * hh_l
    interp_r = w00 * ll_r + w10 * hl_r + w01 * lh_r + w11 * hh_r
    return interp_l, interp_r
