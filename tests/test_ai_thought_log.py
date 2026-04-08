"""test_ai_thought_log.py — Tests for the AI Thought Log watcher."""
from __future__ import annotations

import json
import os
import tempfile

from claudio.watcher import ThoughtLogger, thought_context


def test_log_thought_and_outcome():
    """Should record a thought and its outcome."""
    logger = ThoughtLogger(log_path=os.devnull)
    logger.log_thought("scanner", "Clap detected at -12dBFS", action="run_rt60")
    assert logger.pending_count == 1

    logger.log_outcome("scanner", "RT60 = 0.42s", success=True, duration_ms=5.2)
    assert logger.pending_count == 0

    recent = logger.get_recent(1)
    assert len(recent) == 1
    assert recent[0].agent_id == "scanner"
    assert recent[0].outcome == "RT60 = 0.42s"
    assert recent[0].success is True


def test_thought_context_success():
    """Context manager should log thought + outcome on success."""
    logger = ThoughtLogger(log_path=os.devnull)
    with thought_context(logger, "classifier", "Peak energy > threshold"):
        x = 1 + 1  # noqa: F841

    assert logger.pending_count == 0
    recent = logger.get_recent(1)
    assert recent[0].success is True
    assert recent[0].duration_ms is not None


def test_thought_context_failure():
    """Context manager should log failure on exception."""
    logger = ThoughtLogger(log_path=os.devnull)
    try:
        with thought_context(logger, "classifier", "Will fail intentionally"):
            raise ValueError("test error")
    except ValueError:
        pass

    assert logger.pending_count == 0
    failures = logger.get_failures()
    assert len(failures) == 1
    assert failures[0].success is False
    assert "test error" in failures[0].outcome


def test_jsonl_persistence():
    """Should flush completed entries to JSONL file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir="."
    ) as f:
        path = f.name

    try:
        logger = ThoughtLogger(log_path=path)
        logger.log_thought("agent_a", "Testing persistence")
        logger.log_outcome("agent_a", "persisted", success=True)

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["agent_id"] == "agent_a"
        assert entry["success"] is True
    finally:
        os.unlink(path)


def test_get_failures_empty():
    """Should return empty when no failures exist."""
    logger = ThoughtLogger(log_path=os.devnull)
    logger.log_thought("agent", "All good")
    logger.log_outcome("agent", "OK", success=True)
    assert logger.get_failures() == []
