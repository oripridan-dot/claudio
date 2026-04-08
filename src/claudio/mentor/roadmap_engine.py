"""
roadmap_engine.py — Progressive Disclosure Roadmap Engine

Guides users through the professional production pipeline from setup → mastering,
using progressive UI disclosure: only showing tools relevant to the current phase.

Architecture:
  - ProductionPhase: enum of stages (setup → tracking → mixing → mastering → release)
  - PhaseGate: conditions that must be met before advancing
  - RoadmapEngine: state machine tracking user's journey + triggering UI changes
  - PhaseChecklist: per-phase tasks with completion tracking

Phase configurations (checklist items, UI panels) are defined in roadmap_phases.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .knowledge_base import ProductionPhase


class PhaseStatus(Enum):
    LOCKED = "locked"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class ChecklistItem:
    """A single task within a production phase."""
    item_id: str
    label: str
    description: str
    is_automated: bool          # True = Claudio can auto-detect completion
    completed: bool = False
    auto_detection_key: str = ""  # key from AI detections that marks this done
    mentor_tip_id: str = ""     # linked mentor tip to surface when working on this


@dataclass
class PhaseConfig:
    """Configuration for a single production phase."""
    phase: ProductionPhase
    title: str
    subtitle: str
    status: PhaseStatus = PhaseStatus.LOCKED
    checklist: list[ChecklistItem] = field(default_factory=list)
    ui_panels_visible: list[str] = field(default_factory=list)
    ui_panels_hidden: list[str] = field(default_factory=list)
    completion_percentage: float = 0.0


@dataclass
class RoadmapState:
    """Full roadmap state for serialization to the UI."""
    current_phase: ProductionPhase
    phases: list[PhaseConfig]
    overall_progress: float  # 0-100%


# ─── Roadmap Engine ──────────────────────────────────────────────────────────

class RoadmapEngine:
    """
    State machine that tracks a user's progression through the production pipeline.

    Responds to detection events from the intelligence layer to auto-complete
    checklist items, advance phases, and control which UI panels are visible.
    """

    def __init__(self) -> None:
        from .roadmap_phases import (
            build_mastering_phase,
            build_mixing_phase,
            build_setup_phase,
            build_tracking_phase,
        )

        self._phases = [
            build_setup_phase(),
            build_tracking_phase(),
            build_mixing_phase(),
            build_mastering_phase(),
        ]
        self._current_idx = 0
        self._phases[0].status = PhaseStatus.ACTIVE

    @property
    def current_phase(self) -> PhaseConfig:
        return self._phases[self._current_idx]

    @property
    def state(self) -> RoadmapState:
        total_items = sum(len(p.checklist) for p in self._phases)
        completed_items = sum(
            sum(1 for item in p.checklist if item.completed) for p in self._phases
        )
        return RoadmapState(
            current_phase=self._phases[self._current_idx].phase,
            phases=self._phases,
            overall_progress=round((completed_items / max(1, total_items)) * 100, 1),
        )

    def process_detection(self, detection_key: str) -> list[str]:
        """
        Process a detection event. Auto-completes matching checklist items.
        Returns list of newly completed item IDs.
        """
        completed: list[str] = []
        for phase in self._phases:
            for item in phase.checklist:
                if item.auto_detection_key == detection_key and not item.completed:
                    item.completed = True
                    completed.append(item.item_id)
        self._update_status()
        return completed

    def complete_item(self, item_id: str) -> bool:
        """Manually mark a checklist item as completed."""
        for phase in self._phases:
            for item in phase.checklist:
                if item.item_id == item_id:
                    item.completed = True
                    self._update_status()
                    return True
        return False

    def advance_phase(self) -> ProductionPhase | None:
        """Manually advance to the next phase (skip gate)."""
        if self._current_idx < len(self._phases) - 1:
            self._phases[self._current_idx].status = PhaseStatus.COMPLETED
            self._current_idx += 1
            self._phases[self._current_idx].status = PhaseStatus.ACTIVE
            return self._phases[self._current_idx].phase
        return None

    def get_visible_panels(self) -> list[str]:
        """Return the list of UI panel IDs that should be visible in the current phase."""
        return self.current_phase.ui_panels_visible

    def get_hidden_panels(self) -> list[str]:
        """Return the list of UI panel IDs that should be hidden in the current phase."""
        return self.current_phase.ui_panels_hidden

    def _update_status(self) -> None:
        """Recalculate phase completion percentages and auto-advance if ready."""
        for phase in self._phases:
            total = len(phase.checklist)
            done = sum(1 for item in phase.checklist if item.completed)
            phase.completion_percentage = round((done / max(1, total)) * 100, 1)

        # Auto-advance if current phase is 100% complete
        current = self._phases[self._current_idx]
        if current.completion_percentage >= 100:
            current.status = PhaseStatus.COMPLETED
            if self._current_idx < len(self._phases) - 1:
                self._current_idx += 1
                self._phases[self._current_idx].status = PhaseStatus.ACTIVE
