"""
roadmap_phases.py — Default Phase Configurations for the Production Roadmap

Pure data definitions for each production phase's checklist items and
UI panel visibility. Extracted from roadmap_engine.py for 300-line compliance.
"""
from __future__ import annotations

from .knowledge_base import ProductionPhase
from .roadmap_engine import ChecklistItem, PhaseConfig, PhaseStatus

# ─── Default Phase Configurations ─────────────────────────────────────────────

def build_setup_phase() -> PhaseConfig:
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
                label="Check Microphone Placement",
                description="Verify microphone positioning relative to the source.",
                is_automated=True,
                auto_detection_key="mic_placement_validated",
                mentor_tip_id="STEREO_MIC_TECHNIQUE",
            ),
            ChecklistItem(
                item_id="setup_phase_check",
                label="Multi-Mic Phase Check",
                description="If using multiple mics, check for phase cancellation.",
                is_automated=True,
                auto_detection_key="phase_checked",
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


def build_tracking_phase() -> PhaseConfig:
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


def build_mixing_phase() -> PhaseConfig:
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


def build_mastering_phase() -> PhaseConfig:
    return PhaseConfig(
        phase=ProductionPhase.MASTERING,
        title="Mastering & Quality Control",
        subtitle="Final polish — loudness, clarity, and format compliance.",
        checklist=[
            ChecklistItem(
                item_id="master_loudness",
                label="Loudness Target",
                description=(
                    "Spotify -14, Apple Music -16, YouTube -14, Amazon -14 LUFS. "
                    "Set true peak ceiling to -1.0 dBTP (AES77-2023)."
                ),
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
                item_id="master_delivery_formats",
                label="Delivery Format QC",
                description=(
                    "Validate against lossy codec artefacts (AAC, Opus, Vorbis). "
                    "Check for inter-sample clipping introduced by encoding."
                ),
                is_automated=False,
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
