"""
gesture_heuristics.py — Static Gesture Detection Heuristics

Low-level finger/hand pose detection rules for MediaPipe landmarks.
Extracted from gesture_classifier.py for 300-line compliance.
"""

from __future__ import annotations

import numpy as np


def is_open_palm(lm: np.ndarray) -> bool:
    """
    Open palm: all four finger tips (8,12,16,20) above their MCPs (5,9,13,17).
    MediaPipe y=0 is top, so "above" means lower y value.
    """
    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        if lm[tip, 1] >= lm[mcp, 1]:
            return False
    return True


def is_fist(lm: np.ndarray) -> bool:
    """
    Fist: all four finger tips (8,12,16,20) below their PIPs (6,10,14,18).
    (Curled fingers: tip y > pip y in image space.)
    """
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if lm[tip, 1] <= lm[pip, 1]:
            return False
    return True
