"""test_sofa_loader.py — Unit tests for the SOFA file loader."""
from __future__ import annotations

import numpy as np

from claudio.sofa_loader import SOFADatabase, SOFAMetadata, _create_stub_database


def test_stub_database_has_measurements():
    """Stub database should have procedural measurements at a coarse grid."""
    db = _create_stub_database()
    assert db.metadata.num_measurements > 0
    assert len(db.source_positions) == db.metadata.num_measurements
    assert db.ir_data.shape[0] == db.metadata.num_measurements


def test_stub_hrir_lookup():
    """Should return valid HRIR pair for any azimuth/elevation."""
    db = _create_stub_database()
    hrir_l, hrir_r = db.get_hrir(45.0, 0.0)
    assert hrir_l.shape == (256,)
    assert hrir_r.shape == (256,)
    assert hrir_l.dtype == np.float32
    assert np.any(hrir_l != 0)  # not all zeros


def test_hrir_resize():
    """get_hrir should resize to target_len."""
    db = _create_stub_database()
    hrir_l, hrir_r = db.get_hrir(0.0, 0.0, target_len=128)
    assert hrir_l.shape == (128,)
    assert hrir_r.shape == (128,)


def test_nearest_neighbour_lookup():
    """A query at a non-grid angle should return the nearest measurement."""
    db = _create_stub_database()
    # Query at 47° — should find nearest grid point
    hrir_l, hrir_r = db.get_hrir(47.0, 3.0)
    assert hrir_l.shape == (256,)


def test_grid_resolution():
    """Stub database should report a reasonable grid resolution."""
    db = _create_stub_database()
    res = db.grid_resolution_deg
    assert 5.0 <= res <= 30.0


def test_direct_database_creation():
    """Should be able to create a database from raw arrays."""
    positions = np.array([[0.0, 0.0, 1.0], [90.0, 0.0, 1.0]], dtype=np.float64)
    ir_data = np.random.randn(2, 2, 64).astype(np.float32)
    db = SOFADatabase(
        metadata=SOFAMetadata(num_measurements=2, hrir_length=64),
        source_positions=positions,
        ir_data=ir_data,
    )
    hrir_l, hrir_r = db.get_hrir(0.0, 0.0)
    assert hrir_l.shape == (64,)
