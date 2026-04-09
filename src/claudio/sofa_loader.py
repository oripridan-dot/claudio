"""
sofa_loader.py — SOFA File Loader for Personalised HRTF Profiles

Loads AES69-2022 SOFA (.sofa) files containing measured HRTF datasets
(e.g. from CIPIC, LISTEN, or 3D Tune-In) and provides them as drop-in
replacements for the procedural HRIRs in hrtf_data.py.

SOFA format:
  - HDF5-based container (NetCDF-4 subset)
  - Convention: SimpleFreeFieldHRIR
  - Required datasets: Data.IR, Data.SamplingRate, SourcePosition
  - SourcePosition columns: [azimuth, elevation, distance] (degrees/metres)

Usage:
    from claudio.sofa_loader import SOFADatabase, load_sofa

    db = load_sofa("assets/hrtf/Subject_003.sofa")
    hrir_l, hrir_r = db.get_hrir(azimuth_deg=45.0, elevation_deg=0.0)

Dependencies:
    - h5py (optional, for reading real SOFA files)
    - Falls back to stub HRIR if h5py is unavailable
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

SOFA_DATASET_PATH = "assets/hrtf"


@dataclass
class SOFAMetadata:
    """Metadata from a SOFA file header."""

    title: str = ""
    database_name: str = ""
    listener_description: str = ""
    source_description: str = ""
    date_created: str = ""
    data_type: str = "FIR"
    sofa_convention: str = "SimpleFreeFieldHRIR"
    sample_rate: float = 48000.0
    num_measurements: int = 0
    hrir_length: int = 256


@dataclass
class SOFADatabase:
    """
    Parsed SOFA HRTF database providing spatial HRIR lookup.

    Stores all measurements indexed by (azimuth, elevation) grid for
    O(1) nearest-neighbour lookup with optional bilinear interpolation.
    """

    metadata: SOFAMetadata
    source_positions: np.ndarray  # (N, 3): azimuth, elevation, distance
    ir_data: np.ndarray  # (N, 2, hrir_len): left+right HRIRs
    _grid_index: dict[tuple[int, int], int] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Build spatial grid index for fast lookup."""
        for i in range(len(self.source_positions)):
            az = int(round(self.source_positions[i, 0]))
            el = int(round(self.source_positions[i, 1]))
            self._grid_index[(az % 360, el)] = i

    def get_hrir(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        target_len: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Look up the closest measured HRIR pair for a given direction.

        Returns (left_hrir, right_hrir) as float32 numpy arrays.
        If target_len is specified and differs from the stored length,
        the HRIR is zero-padded or truncated.
        """
        az_key = int(round(azimuth_deg)) % 360
        el_key = int(round(elevation_deg))

        # Exact match
        idx = self._grid_index.get((az_key, el_key))

        # Nearest-neighbour fallback
        if idx is None:
            idx = self._find_nearest(azimuth_deg, elevation_deg)

        hrir_l = self.ir_data[idx, 0, :].astype(np.float32)
        hrir_r = self.ir_data[idx, 1, :].astype(np.float32)

        if target_len is not None and target_len != len(hrir_l):
            hrir_l = _resize_hrir(hrir_l, target_len)
            hrir_r = _resize_hrir(hrir_r, target_len)

        return hrir_l, hrir_r

    def _find_nearest(self, az_deg: float, el_deg: float) -> int:
        """Find the index of the closest measurement to the query direction."""
        az_rad = math.radians(az_deg)
        el_rad = math.radians(el_deg)
        query = np.array(
            [
                math.cos(el_rad) * math.cos(az_rad),
                math.cos(el_rad) * math.sin(az_rad),
                math.sin(el_rad),
            ]
        )

        best_idx = 0
        best_dot = -2.0
        for i in range(len(self.source_positions)):
            sp_az = math.radians(self.source_positions[i, 0])
            sp_el = math.radians(self.source_positions[i, 1])
            sp = np.array(
                [
                    math.cos(sp_el) * math.cos(sp_az),
                    math.cos(sp_el) * math.sin(sp_az),
                    math.sin(sp_el),
                ]
            )
            dot = float(np.dot(query, sp))
            if dot > best_dot:
                best_dot = dot
                best_idx = i

        return best_idx

    @property
    def grid_resolution_deg(self) -> float:
        """Estimate the average grid resolution from source positions."""
        if len(self.source_positions) < 2:
            return 5.0
        azimuths = np.unique(np.round(self.source_positions[:, 0]))
        if len(azimuths) > 1:
            diffs = np.diff(np.sort(azimuths))
            return float(np.median(diffs))
        return 5.0


def _resize_hrir(hrir: np.ndarray, target_len: int) -> np.ndarray:
    """Resize HRIR to target_len via truncation or zero-padding."""
    if len(hrir) >= target_len:
        return hrir[:target_len]
    padded = np.zeros(target_len, dtype=np.float32)
    padded[: len(hrir)] = hrir
    return padded


# ─── SOFA File Loading ───────────────────────────────────────────────────────


def load_sofa(filepath: str) -> SOFADatabase:
    """
    Load a SOFA (.sofa) file and return a SOFADatabase.

    Requires h5py for reading HDF5-based SOFA files. Falls back to a
    stub database if h5py is unavailable or the file cannot be read.
    """
    if not os.path.exists(filepath):
        logger.warning("SOFA file not found: %s — using procedural fallback", filepath)
        return _create_stub_database()

    try:
        import h5py
    except ImportError:
        logger.warning("h5py not installed — cannot read SOFA files. Using stub.")
        return _create_stub_database()

    try:
        return _load_sofa_h5py(filepath, h5py)
    except Exception:
        logger.exception("Failed to load SOFA file: %s", filepath)
        return _create_stub_database()


def _load_sofa_h5py(filepath: str, h5py: object) -> SOFADatabase:
    """Parse a SOFA file using h5py."""
    with h5py.File(filepath, "r") as f:  # type: ignore[union-attr]
        # Read metadata
        meta = SOFAMetadata()
        meta.title = _read_attr(f, "Title")
        meta.database_name = _read_attr(f, "DatabaseName")
        meta.listener_description = _read_attr(f, "ListenerDescription")
        meta.date_created = _read_attr(f, "DateCreated")
        meta.sofa_convention = _read_attr(f, "SOFAConventions", "SimpleFreeFieldHRIR")

        # Read sampling rate
        sr_data = f.get("Data.SamplingRate")
        if sr_data is not None:
            meta.sample_rate = float(sr_data[()])

        # Read source positions: (N, 3) — azimuth, elevation, distance
        sp_data = f.get("SourcePosition")
        if sp_data is None:
            raise ValueError("SOFA file missing SourcePosition dataset")
        source_positions = np.array(sp_data, dtype=np.float64)

        # Read IR data: (N, R, L) where R=2 (left,right), L=hrir_length
        ir_data_raw = f.get("Data.IR")
        if ir_data_raw is None:
            raise ValueError("SOFA file missing Data.IR dataset")
        ir_data = np.array(ir_data_raw, dtype=np.float32)

        meta.num_measurements = ir_data.shape[0]
        meta.hrir_length = ir_data.shape[2] if ir_data.ndim == 3 else 256

    logger.info(
        "Loaded SOFA: %s — %d measurements, %d samples, %.0f Hz",
        filepath,
        meta.num_measurements,
        meta.hrir_length,
        meta.sample_rate,
    )
    return SOFADatabase(
        metadata=meta,
        source_positions=source_positions,
        ir_data=ir_data,
    )


def _read_attr(f: object, key: str, default: str = "") -> str:
    """Safely read an HDF5 attribute as string."""
    attrs = getattr(f, "attrs", {})
    val = attrs.get(key, default)
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val) if val else default


def _create_stub_database() -> SOFADatabase:
    """Create a minimal stub database for fallback (no real SOFA data)."""
    from claudio.hrtf_data import get_hrir

    # Generate procedural HRIRs at a coarse grid to populate the stub
    hrir_len = 256
    azimuths = range(0, 360, 15)
    elevations = range(-40, 91, 10)
    positions = []
    ir_pairs = []

    for az in azimuths:
        for el in elevations:
            hrir_l, hrir_r = get_hrir(float(az), float(el), hrir_len)
            positions.append([float(az), float(el), 1.0])
            ir_pairs.append(np.stack([hrir_l, hrir_r]))

    return SOFADatabase(
        metadata=SOFAMetadata(
            title="Procedural Stub",
            database_name="claudio-procedural",
            sample_rate=192_000,
            num_measurements=len(positions),
            hrir_length=hrir_len,
        ),
        source_positions=np.array(positions, dtype=np.float64),
        ir_data=np.array(ir_pairs, dtype=np.float32),
    )


def list_sofa_files(directory: str = SOFA_DATASET_PATH) -> list[str]:
    """List all .sofa files in the HRTF asset directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".sofa"))
