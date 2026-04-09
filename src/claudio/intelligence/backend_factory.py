"""
backend_factory.py — Smart Backend Selection

Factory for selecting the optimal neural audio classification backend
based on benchmarking results and system capabilities.

Benchmark Summary (physics-based audio, 9 instruments):
  ┌─────────┬─────────────┬────────────┬────────────────────────────────────┐
  │ Backend │ Avg Latency │ Avg Conf   │ Key Strength                       │
  ├─────────┼─────────────┼────────────┼────────────────────────────────────┤
  │ PANNs   │    73ms     │   61%      │ Fastest, generic "Music" detection │
  │ CLAP    │   722ms     │   48%      │ Best instrument discrimination     │
  │ AST     │   658ms     │   36%      │ Moderate accuracy                  │
  └─────────┴─────────────┴────────────┴────────────────────────────────────┘

  Winner: CLAP for production (best instrument identification)
  Runner-up: PANNs for real-time fallback (10x faster)
"""

from __future__ import annotations

from enum import Enum

from .classifier_backend import AudioClassifierBackend


class BackendStrategy(Enum):
    """Backend selection strategy."""

    QUALITY = "quality"  # CLAP — best instrument discrimination
    SPEED = "speed"  # PANNs — fastest inference
    AUTO = "auto"  # Try CLAP, fallback to PANNs
    NONE = "none"  # No neural backend (heuristic only)


def create_backend(
    strategy: BackendStrategy = BackendStrategy.AUTO,
    device: str = "cpu",
) -> AudioClassifierBackend | None:
    """
    Create the optimal neural backend based on strategy.

    Args:
        strategy: Selection strategy
        device: Compute device ("cpu", "cuda", "mps")

    Returns:
        An initialized AudioClassifierBackend, or None if unavailable.
    """
    if strategy == BackendStrategy.NONE:
        return None

    if strategy == BackendStrategy.QUALITY:
        return _try_clap(device) or _try_panns(device)

    if strategy == BackendStrategy.SPEED:
        return _try_panns(device) or _try_clap(device)

    # AUTO — try CLAP first, then PANNs
    return _try_clap(device) or _try_panns(device)


def _try_clap(device: str) -> AudioClassifierBackend | None:
    """Attempt to load CLAP backend."""
    try:
        from .backend_clap import CLAPBackend

        backend = CLAPBackend(device=device)
        backend.load_model()
        return backend
    except Exception as e:
        print(f"[BackendFactory] CLAP unavailable: {e}")
        return None


def _try_panns(device: str) -> AudioClassifierBackend | None:
    """Attempt to load PANNs backend."""
    try:
        from .backend_panns import PANNsBackend

        backend = PANNsBackend(device=device)
        backend.load_model()
        return backend
    except Exception as e:
        print(f"[BackendFactory] PANNs unavailable: {e}")
        return None
