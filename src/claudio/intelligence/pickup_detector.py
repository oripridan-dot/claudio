"""
pickup_detector.py — Electric Guitar Pickup Type Classifier

Classifies pickup type (single coil, humbucker, P90, piezo, split coil)
from spectral characteristics: centroid, flatness, rolloff.

Extracted from instrument_classifier.py for single-responsibility compliance.
"""

from __future__ import annotations

from enum import Enum

from .spectral_extractor import SpectralFingerprint


class PickupType(Enum):
    SINGLE_COIL = "single_coil"
    HUMBUCKER = "humbucker"
    SPLIT_COIL = "split_coil"  # Fender P-Bass style (2 halves, hum-cancelling)
    P90 = "p90"
    PIEZO = "piezo"
    ACTIVE = "active"
    UNKNOWN = "unknown"


class PickupDetector:
    """Classifies electric guitar pickup type from spectral characteristics."""

    # Single-coil: bright (high centroid), narrow bandwidth, prominent 3-6kHz
    # Humbucker: darker, wider, stronger mids, weaker highs
    # P90: midrange bark, moderate brightness
    # Piezo: ultra-bright, thin, high flatness

    def classify(self, fingerprint: SpectralFingerprint) -> tuple[PickupType, float]:
        centroid = fingerprint.spectral_centroid_hz
        flatness = fingerprint.spectral_flatness
        rolloff = fingerprint.spectral_rolloff_hz

        scores: dict[PickupType, float] = {}

        # Single coil: bright, clear
        sc_score = 0.0
        if centroid > 2500:
            sc_score += 0.4
        if rolloff > 8000:
            sc_score += 0.3
        if flatness < 0.3:
            sc_score += 0.3
        scores[PickupType.SINGLE_COIL] = sc_score

        # Humbucker: darker, thicker
        hb_score = 0.0
        if centroid < 2200:
            hb_score += 0.4
        if rolloff < 7000:
            hb_score += 0.3
        if flatness < 0.25:
            hb_score += 0.3
        scores[PickupType.HUMBUCKER] = hb_score

        # P90: midrange presence
        p90_score = 0.0
        if 1800 < centroid < 3000:
            p90_score += 0.5
        if 5000 < rolloff < 9000:
            p90_score += 0.3
        if flatness < 0.3:
            p90_score += 0.2
        scores[PickupType.P90] = p90_score

        # Piezo: very bright, thin
        pz_score = 0.0
        if centroid > 3500:
            pz_score += 0.4
        if flatness > 0.35:
            pz_score += 0.4
        if rolloff > 10000:
            pz_score += 0.2
        scores[PickupType.PIEZO] = pz_score

        best = max(scores, key=scores.__getitem__)
        return best, scores[best]
