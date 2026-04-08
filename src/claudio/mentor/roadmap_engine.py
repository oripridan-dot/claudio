"""
roadmap_engine.py — Progressive Disclosure Roadmap Engine

Guides users through the professional production pipeline from setup → mastering,
using progressive UI disclosure: only showing tools relevant to the current phase.

Architecture:
  - ProductionPhase: enum of stages (setup → tracking → mixing → mastering → release)
  - PhaseGate: conditions that must be met before advancing
  - RoadmapEngine: state machine tracking user's journey + triggering UI changes
  - PhaseChecklist: per-phase tasks with completion tracking
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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


# ─── Default Phase Configurations ─────────────────────────────────────────────

def _build_setup_phase() -> PhaseConfig:
    return PhaseConfig(
        phase=ProductionPhase.SETUP,
        title="Room Setup & Calibration",
        subtitle="Perfect the analog signal before it hits the converters.",
        status=PhaseStatus.ACTIVE,
        checklist=[
            ChecklistItem(
                item_id="setup_room_scan",
                label="Complete Room Scan",
                description="Claudio analyzes your room acoustics with a clap or sweep.",
                is_automated=True,
                auto_detection_key="room_scan_complete",
            ),
            ChecklistItem(
                item_id="setup_gain_staging",
                label="Optimize Gain Staging",
                description="Set preamp levels with 6dB headroom on peaks.",
                is_automated=True,
                auto_detection_key="gain_staging_optimal",
                mentor_tip_id="PREAMP_CLIPPING",
            ),
            ChecklistItem(
                item_id="setup_mic_placement",
                label="Position Microphones",
                description="Place mics for optimal source capture.",
                is_automated=True,
                auto_detection_key="mic_placement_validated",
                mentor_tip_id="ACOUSTIC_GUITAR_SOUNDHOLE",
            ),
            ChecklistItem(
                item_id="setup_phase_check",
                label="Verify Phase Alignment",
                description="Check polarity on all multi-mic sources.",
                is_automated=True,
                auto_detection_key="phase_aligned",
                mentor_tip_id="DRUM_SNARE_PHASE_180",
            ),
            ChecklistItem(
                item_id="setup_sweet_spot",
                label="Calibrate Listening Position",
                description="Ensure you're in the sweet spot for monitoring.",
                is_automated=True,
                auto_detection_key="sweet_spot_calibrated",
                mentor_tip_id="MIXING_OFF_AXIS",
            ),
        ],
        ui_panels_visible=[
            "room_scanner", "phase_meter", "gain_meter",
            "instrument_detector", "mic_placement_guide", "sweet_spot_hud",
        ],
        ui_panels_hidden=[
            "mix_bus", "mastering_chain", "lufs_meter",
            "stereo_width", "release_checklist",
        ],
    )


def _build_tracking_phase() -> PhaseConfig:
    return PhaseConfig(
        phase=ProductionPhase.TRACKING,
        title="Recording & Performance",
        subtitle="Capture the best possible human performance.",
        checklist=[
            ChecklistItem(
                item_id="tracking_first_take",
                label="Record First Take",
                description="Get the initial performance recorded.",
                is_automated=True,
                auto_detection_key="recording_captured",
            ),
            ChecklistItem(
                item_id="tracking_timing",
                label="Timing Check",
                description="Review groove consistency and pocket alignment.",
                is_automated=True,
                auto_detection_key="timing_acceptable",
                mentor_tip_id="TAKE_ENERGY_DROP",
            ),
            ChecklistItem(
                item_id="tracking_dynamics",
                label="Dynamics Review",
                description="Ensure healthy dynamic range in performance.",
                is_automated=True,
                auto_detection_key="dynamics_healthy",
            ),
            ChecklistItem(
                item_id="tracking_tone",
                label="Tone Optimization",
                description="Optimize instrument tone at the source.",
                is_automated=True,
                auto_detection_key="tone_optimized",
                mentor_tip_id="GUITAR_TONE_KNOB",
            ),
        ],
        ui_panels_visible=[
            "waveform", "performance_coach", "pocket_radar",
            "instrument_detector", "mentor_overlay",
        ],
        ui_panels_hidden=[
            "room_scanner", "mix_bus", "mastering_chain",
            "lufs_meter", "release_checklist",
        ],
    )


def _build_mixing_phase() -> PhaseConfig:
    return PhaseConfig(
        phase=ProductionPhase.MIXING,
        title="Mixing & Balance",
        subtitle="Carve space, manage masking, create dimension.",
        checklist=[
            ChecklistItem(
                item_id="mix_frequency_balance",
                label="Resolve Frequency Collisions",
                description="Fix masking zones between competing sources.",
                is_automated=True,
                auto_detection_key="collisions_resolved",
                mentor_tip_id="KICK_BASS_MUD",
            ),
            ChecklistItem(
                item_id="mix_stereo_image",
                label="Stereo Image Check",
                description="Verify panning and stereo width.",
                is_automated=True,
                auto_detection_key="stereo_balanced",
            ),
            ChecklistItem(
                item_id="mix_mono_compat",
                label="Mono Compatibility Test",
                description="Ensure critical elements survive mono sum.",
                is_automated=True,
                auto_detection_key="mono_compatible",
                mentor_tip_id="STEREO_MONO_COLLAPSE",
            ),
            ChecklistItem(
                item_id="mix_dynamics",
                label="Dynamic Processing",
                description="Apply compression and expansion where needed.",
                is_automated=False,
            ),
        ],
        ui_panels_visible=[
            "spectrum_analyzer", "topographic_freq_map", "stereo_width",
            "phase_meter", "mix_bus", "mentor_overlay", "sweet_spot_hud",
        ],
        ui_panels_hidden=[
            "room_scanner", "mic_placement_guide", "mastering_chain",
            "release_checklist",
        ],
    )


def _build_mastering_phase() -> PhaseConfig:
    return PhaseConfig(
        phase=ProductionPhase.MASTERING,
        title="Mastering & Quality Control",
        subtitle="Final polish — loudness, clarity, and format compliance.",
        checklist=[
            ChecklistItem(
                item_id="master_loudness",
                label="Loudness Target",
                description="Hit -14 LUFS integrated for streaming platforms.",
                is_automated=True,
                auto_detection_key="lufs_target_met",
                mentor_tip_id="MASTERING_OVER_LIMITING",
            ),
            ChecklistItem(
                item_id="master_true_peak",
                label="True Peak Check",
                description="Ensure no inter-sample peaks above -1 dBTP.",
                is_automated=True,
                auto_detection_key="true_peak_safe",
            ),
            ChecklistItem(
                item_id="master_mono_check",
                label="Final Mono Check",
                description="Last mono compatibility verification.",
                is_automated=True,
                auto_detection_key="final_mono_ok",
            ),
            ChecklistItem(
                item_id="master_metadata",
                label="Metadata & Tags",
                description="Verify ISRC, artist name, album art, genre tags.",
                is_automated=False,
            ),
        ],
        ui_panels_visible=[
            "lufs_meter", "true_peak_meter", "stereo_width",
            "mastering_chain", "mentor_overlay",
        ],
        ui_panels_hidden=[
            "room_scanner", "mic_placement_guide", "pocket_radar",
            "performance_coach",
        ],
    )


# ─── Roadmap Engine ──────────────────────────────────────────────────────────

class RoadmapEngine:
    """
    State machine that tracks a user's progression through the production pipeline.

    Responds to detection events from the intelligence layer to auto-complete
    checklist items, advance phases, and control which UI panels are visible.
    """

    def __init__(self) -> None:
        self._phases = [
            _build_setup_phase(),
            _build_tracking_phase(),
            _build_mixing_phase(),
            _build_mastering_phase(),
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

    def advance_phase(self) -> Optional[ProductionPhase]:
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
