"""
ai_thought_log.py — AI Thought Log Watcher for Autonomous Operations

Implements the AI_THOUGHT_LOG pattern from the Autonomous AI Operations
framework. Before any autonomous action, agents must log their reasoning.

Usage:
    from claudio.watcher.ai_thought_log import ThoughtLogger, thought_context

    logger = ThoughtLogger()

    # Context manager usage:
    with thought_context(logger, "instrument_classifier",
                         "Classifying audio because peak energy > -20dBFS"):
        result = classifier.classify(audio_buffer)

    # Direct usage:
    logger.log_thought("room_scanner", "Running RT60 because clap detected",
                       context={"peak_db": -12.3, "trigger": "transient"})
    scanner.analyze(buffer)
    logger.log_outcome("room_scanner", "RT60 = 0.42s, room classified as studio")
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Default log location inside the workspace
DEFAULT_LOG_PATH = "AI_THOUGHT_LOG.jsonl"
MAX_LOG_ENTRIES = 10_000  # Rotate after this many entries


@dataclass
class ThoughtEntry:
    """A single pre-action reasoning record."""
    timestamp: str
    agent_id: str
    action: str
    reasoning: str
    context: dict[str, Any] = field(default_factory=dict)
    outcome: str | None = None
    duration_ms: float | None = None
    success: bool | None = None


class ThoughtLogger:
    """
    Flight data recorder for autonomous AI agent actions.

    Rule 4 from the Autonomous AI Operations framework:
    "Before writing any code, the agent writes a paragraph explaining
    WHY it chose this approach."

    This logger captures:
    1. Pre-action reasoning (the "thought")
    2. Context data (inputs, triggers, state)
    3. Post-action outcome (result, success/failure)
    4. Duration for performance tracking

    Log format: JSONL (one JSON object per line) for streaming reads.
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH) -> None:
        self._log_path = log_path
        self._entries: list[ThoughtEntry] = []
        self._pending: dict[str, ThoughtEntry] = {}  # agent_id → in-progress entry

    def log_thought(
        self,
        agent_id: str,
        reasoning: str,
        *,
        action: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a pre-action thought — MUST be called before the action executes.

        Args:
            agent_id: Which agent/module is acting (e.g. "room_scanner")
            reasoning: WHY this action is being taken
            action: WHAT action will be performed (optional, for clarity)
            context: supporting data (trigger values, thresholds, etc.)
        """
        entry = ThoughtEntry(
            timestamp=datetime.now(tz=UTC).isoformat(),
            agent_id=agent_id,
            action=action,
            reasoning=reasoning,
            context=context or {},
        )
        self._pending[agent_id] = entry
        self._entries.append(entry)
        logger.debug("THOUGHT [%s]: %s", agent_id, reasoning)

    def log_outcome(
        self,
        agent_id: str,
        outcome: str,
        *,
        success: bool = True,
        duration_ms: float | None = None,
    ) -> None:
        """
        Log the outcome of a previously logged thought.

        Args:
            agent_id: Must match a previous log_thought call
            outcome: What happened (result description)
            success: Whether the action succeeded
            duration_ms: How long the action took
        """
        entry = self._pending.pop(agent_id, None)
        if entry is None:
            logger.warning("log_outcome without prior log_thought for agent: %s", agent_id)
            return

        entry.outcome = outcome
        entry.success = success
        entry.duration_ms = duration_ms
        self._flush_entry(entry)

    def get_recent(self, n: int = 20) -> list[ThoughtEntry]:
        """Return the most recent N thought entries."""
        return self._entries[-n:]

    def get_failures(self, n: int = 10) -> list[ThoughtEntry]:
        """Return the most recent N failed actions."""
        failures = [e for e in self._entries if e.success is False]
        return failures[-n:]

    def _flush_entry(self, entry: ThoughtEntry) -> None:
        """Append a completed entry to the log file (JSONL format)."""
        try:
            os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), default=str) + "\n")
        except OSError:
            logger.exception("Failed to write thought log: %s", self._log_path)

        # Rotate if too large
        if len(self._entries) > MAX_LOG_ENTRIES:
            self._entries = self._entries[-MAX_LOG_ENTRIES // 2:]

    @property
    def pending_count(self) -> int:
        """Number of thoughts logged without an outcome yet."""
        return len(self._pending)


@contextmanager
def thought_context(
    thought_logger: ThoughtLogger,
    agent_id: str,
    reasoning: str,
    *,
    action: str = "",
    context: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    """
    Context manager that logs a thought before and outcome after an action.

    Usage:
        with thought_context(logger, "classifier", "Audio peak > threshold"):
            result = classifier.classify(buffer)
    """
    thought_logger.log_thought(agent_id, reasoning, action=action, context=context)
    t0 = time.perf_counter()
    try:
        yield
        duration_ms = (time.perf_counter() - t0) * 1000
        thought_logger.log_outcome(
            agent_id, "completed", success=True, duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        thought_logger.log_outcome(
            agent_id, f"FAILED: {e!r}", success=False, duration_ms=duration_ms,
        )
        raise
