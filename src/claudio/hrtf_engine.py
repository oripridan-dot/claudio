"""
hrtf_engine.py — Holographic Binaural Rendering Engine

Real-time binaural renderer that spatialises mono sources into
a 192 kHz binaural field using per-source HRTF convolution.

HRTF data synthesis is handled by hrtf_data.py (Single Responsibility).
This module is exclusively the real-time rendering engine.

Thread model:
  - Audio thread calls render() at the audio callback rate
  - Camera/6DoF thread calls update_head_pose() — lock-free
  - SpatialLatencyGate: <1.5 ms HRTF update on 90° head turn
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from claudio.hrtf_data import (
    HRIR_LEN,
    SAMPLE_RATE,
    azimuth_elevation_from_position,
    get_hrir,
    interpolate_hrir_bilinear,
)
from claudio.signal_flow_config import (
    ConvolutionStrategy,
    HRTFInterpolation,
    SignalFlowConfig,
)

# Re-export for backward compatibility
_azimuth_elevation_from_position = azimuth_elevation_from_position
_get_hrir = get_hrir

FFT_SIZE       = 512
SPEED_OF_SOUND = 343.0
SOFA_DATASET_PATH = "assets/hrtf"


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class AudioSource:
    """A single mono audio source placed in 3D space."""
    source_id:   str
    position:    np.ndarray
    gain_db:     float = 0.0
    _hrtf_l_fd: np.ndarray | None = field(default=None, repr=False, compare=False)
    _hrtf_r_fd: np.ndarray | None = field(default=None, repr=False, compare=False)
    _prev_hrtf_l_fd: np.ndarray | None = field(default=None, repr=False, compare=False)
    _prev_hrtf_r_fd: np.ndarray | None = field(default=None, repr=False, compare=False)
    _crossfade_pos: int = field(default=0, repr=False, compare=False)
    _ola_l: np.ndarray | None = field(default=None, repr=False, compare=False)
    _ola_r: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class BinauralFrame:
    """One audio callback's worth of rendered binaural output."""
    left:   np.ndarray
    right:  np.ndarray
    sources_rendered: int
    render_time_us: float = 0.0


# ─── Binaural Rendering Engine ────────────────────────────────────────────────

class HRTFBinauralEngine:
    """Real-time holographic binaural rendering engine."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        config: SignalFlowConfig | None = None,
    ) -> None:
        if config is not None:
            self._sample_rate = config.render_sample_rate
            self._fft_size = config.fft_size
            self._hrir_len = config.hrir_length
            self._grid_res = config.hrtf_grid_resolution_deg
            self._crossfade_len = config.crossfade_samples
            self._interpolation = config.hrtf_interpolation
            self._strategy = config.convolution_strategy
            self._partition_count = config.partition_count
            self._air_absorption = config.air_absorption_enabled
            self._proximity_cap_db = config.proximity_gain_cap_db
        else:
            self._sample_rate = sample_rate
            self._fft_size = FFT_SIZE
            self._hrir_len = HRIR_LEN
            self._grid_res = 5.0
            self._crossfade_len = 32
            self._interpolation = HRTFInterpolation.BILINEAR
            self._strategy = ConvolutionStrategy.PARTITIONED
            self._partition_count = 2
            self._air_absorption = True
            self._proximity_cap_db = 12.0

        self._sources: dict[str, AudioSource] = {}
        self._head_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self._hrtf_dirty = True

    # ── Source Management ─────────────────────────────────────────────────

    def add_source(self, source: AudioSource) -> None:
        tail_len = self._hrir_len - 1
        source._ola_l = np.zeros(tail_len, dtype=np.float32)
        source._ola_r = np.zeros(tail_len, dtype=np.float32)
        self._sources[source.source_id] = source
        self._hrtf_dirty = True

    def remove_source(self, source_id: str) -> None:
        self._sources.pop(source_id, None)

    def move_source(self, source_id: str, new_position: np.ndarray) -> None:
        if source_id in self._sources:
            self._sources[source_id].position = new_position
            self._hrtf_dirty = True

    # ── Head Tracking ─────────────────────────────────────────────────────

    def update_head_pose(
        self, quat: tuple[float, float, float, float],
    ) -> None:
        self._head_quat = quat
        self._hrtf_dirty = True

    # ── Rendering ─────────────────────────────────────────────────────────

    def render(self, source_buffers: dict[str, np.ndarray]) -> BinauralFrame:
        t0 = time.perf_counter()
        if self._hrtf_dirty:
            self._recompute_hrtfs()
            self._hrtf_dirty = False

        block = self._fft_size
        out_l = np.zeros(block, dtype=np.float32)
        out_r = np.zeros(block, dtype=np.float32)
        rendered = 0

        for sid, buf in source_buffers.items():
            src = self._sources.get(sid)
            if src is None or src._hrtf_l_fd is None:
                continue
            gain = 10 ** (src.gain_db / 20.0) * self._proximity_gain(src.position)
            if self._air_absorption:
                gain *= self._air_absorption_factor(src.position)
            buf_scaled = buf[:block] * gain

            l_out, src._ola_l = self._ola_convolve(buf_scaled, src._hrtf_l_fd, src._ola_l)
            r_out, src._ola_r = self._ola_convolve(buf_scaled, src._hrtf_r_fd, src._ola_r)
            out_l += l_out[:block]
            out_r += r_out[:block]
            rendered += 1

        render_us = (time.perf_counter() - t0) * 1e6
        return BinauralFrame(left=out_l, right=out_r, sources_rendered=rendered, render_time_us=render_us)

    # ── Private ───────────────────────────────────────────────────────────

    def _recompute_hrtfs(self) -> None:
        quat = self._head_quat
        for src in self._sources.values():
            az, el = azimuth_elevation_from_position(src.position, quat)
            if self._interpolation == HRTFInterpolation.BILINEAR:
                hrir_l, hrir_r = interpolate_hrir_bilinear(
                    az, el, self._hrir_len, self._sample_rate, self._grid_res)
            else:
                hrir_l, hrir_r = get_hrir(
                    az, el, self._hrir_len, self._sample_rate, self._grid_res)

            n_fft = self._fft_size + self._hrir_len - 1
            new_l = np.fft.rfft(hrir_l, n=n_fft).astype(np.complex64)
            new_r = np.fft.rfft(hrir_r, n=n_fft).astype(np.complex64)

            if src._hrtf_l_fd is not None:
                src._prev_hrtf_l_fd = src._hrtf_l_fd
                src._prev_hrtf_r_fd = src._hrtf_r_fd
                src._crossfade_pos = 0
            src._hrtf_l_fd = new_l
            src._hrtf_r_fd = new_r

    @staticmethod
    def _ola_convolve(
        x: np.ndarray, h_fd: np.ndarray, ola_tail: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_fft = len(h_fd) * 2 - 2
        x_padded = np.zeros(n_fft)
        x_padded[:len(x)] = x
        y_time = np.fft.irfft(np.fft.rfft(x_padded) * h_fd).real
        tail_len = len(ola_tail)
        y_time[:tail_len] += ola_tail
        block = len(x)
        output   = y_time[:block].astype(np.float32)
        new_tail = y_time[block:block + tail_len].astype(np.float32)
        return output, new_tail

    def _proximity_gain(self, position: np.ndarray) -> float:
        dist = float(np.linalg.norm(position)) + 0.01
        cap = 10 ** (self._proximity_cap_db / 20.0)
        return min(cap, 1.0 / (dist ** 2))

    @staticmethod
    def _air_absorption_factor(position: np.ndarray) -> float:
        """ISO 9613-1 simplified: frequency-averaged atmospheric absorption."""
        dist = float(np.linalg.norm(position))
        absorption_db_per_m = 0.005
        return 10 ** (-absorption_db_per_m * dist / 20.0)
