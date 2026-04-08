"""
multimodal_fusion.py — Multimodal Instrument & Environment Fusion Engine

Fuses acoustic signatures (from instrument_classifier) with computer vision
detections (from camera/MediaPipe) to achieve model-level instrument identification.

Architecture:
  - VisionDetection: structured visual object detection result
  - MultimodalFusion: cross-validates audio + vision, boosts confidence
  - InstrumentModelDB: known instrument profiles for matching
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .instrument_classifier import (
    InstrumentClassifier,
    InstrumentDetection,
    InstrumentFamily,
    PickupType,
)

# ─── Vision Detection ────────────────────────────────────────────────────────

class VisualCategory(Enum):
    GUITAR_SOLID_BODY = "guitar_solid_body"
    GUITAR_HOLLOW_BODY = "guitar_hollow_body"
    GUITAR_ACOUSTIC = "guitar_acoustic"
    BASS_ELECTRIC = "bass_electric"
    BASS_ACOUSTIC = "bass_acoustic"
    DRUM_KIT = "drum_kit"
    DRUM_SINGLE = "drum_single"
    MICROPHONE_DYNAMIC = "microphone_dynamic"
    MICROPHONE_CONDENSER = "microphone_condenser"
    MICROPHONE_RIBBON = "microphone_ribbon"
    KEYBOARD = "keyboard"
    AMPLIFIER = "amplifier"
    MONITOR_SPEAKER = "monitor_speaker"
    MIXING_CONSOLE = "mixing_console"
    PERSON = "person"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    x: float          # normalized 0-1
    y: float
    width: float
    height: float


@dataclass
class VisionDetection:
    """A single object detected by computer vision in the camera frame."""
    category: VisualCategory
    confidence: float
    bounding_box: BoundingBox
    brand_text: str = ""          # OCR'd brand name (e.g. "Fender", "Shure")
    model_text: str = ""          # OCR'd model (e.g. "SM58", "Telecaster")
    color_dominant: str = ""      # dominant color of the object
    body_shape: str = ""          # e.g. "single_cutaway", "double_cutaway", "dreadnought"
    headstock_shape: str = ""     # guitar-specific: "inline_6", "3x3", "reverse"


@dataclass
class EnvironmentDetection:
    """Room/environment visual analysis."""
    room_dimensions_estimate: tuple[float, float, float] = (0.0, 0.0, 0.0)  # w, h, d in meters
    reflective_surfaces: list[str] = field(default_factory=list)  # e.g. ["glass_window", "bare_wall"]
    acoustic_treatment: list[str] = field(default_factory=list)   # e.g. ["bass_trap", "diffuser"]
    lighting_quality: str = "unknown"     # "good", "dim", "harsh"
    people_count: int = 0
    head_positions: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, z) per person


@dataclass
class FusedDetection:
    """Cross-validated detection combining audio + vision."""
    instrument: InstrumentDetection
    vision: VisionDetection | None
    environment: EnvironmentDetection | None
    fused_confidence: float           # combined confidence
    model_identification: str         # e.g. "Fender Telecaster (Bridge Pickup)"
    mic_type: str = ""                # detected microphone type
    mic_distance_cm: float = 0.0     # estimated source-to-mic distance
    speaker_positions: list[tuple[float, float, float]] = field(default_factory=list)
    coaching_context: list[str] = field(default_factory=list)


# ─── Instrument Model Database ───────────────────────────────────────────────

@dataclass
class InstrumentModelProfile:
    """Known instrument profile for matching against detections."""
    name: str                         # "Fender Telecaster"
    brand: str                        # "Fender"
    model: str                        # "Telecaster"
    family: InstrumentFamily
    body_shapes: list[str]            # visual identifiers
    headstock_shapes: list[str]
    pickup_configs: list[PickupType]
    spectral_centroid_range: tuple[float, float]  # Hz
    typical_fundamental_range: tuple[float, float]  # Hz
    coaching_notes: list[str]


# Curated database of well-known instruments
INSTRUMENT_MODEL_DB: list[InstrumentModelProfile] = [
    InstrumentModelProfile(
        name="Fender Stratocaster",
        brand="Fender", model="Stratocaster",
        family=InstrumentFamily.GUITAR_ELECTRIC,
        body_shapes=["double_cutaway", "contoured"],
        headstock_shapes=["inline_6"],
        pickup_configs=[PickupType.SINGLE_COIL],
        spectral_centroid_range=(2200, 4500),
        typical_fundamental_range=(82, 1200),
        coaching_notes=[
            "The Strat's single coils are naturally bright. "
            "If the bridge pickup is too piercing, try position 4 (bridge+middle) "
            "for a warmer quack tone.",
        ],
    ),
    InstrumentModelProfile(
        name="Fender Telecaster",
        brand="Fender", model="Telecaster",
        family=InstrumentFamily.GUITAR_ELECTRIC,
        body_shapes=["single_cutaway", "slab"],
        headstock_shapes=["inline_6"],
        pickup_configs=[PickupType.SINGLE_COIL],
        spectral_centroid_range=(2500, 5000),
        typical_fundamental_range=(82, 1200),
        coaching_notes=[
            "The Telecaster bridge pickup has a signature 'twang' — "
            "a sharp 3kHz spike. For recording, try angling a SM57 "
            "1 inch off the center of the speaker cone to tame it.",
        ],
    ),
    InstrumentModelProfile(
        name="Gibson Les Paul",
        brand="Gibson", model="Les Paul",
        family=InstrumentFamily.GUITAR_ELECTRIC,
        body_shapes=["single_cutaway", "arched_top"],
        headstock_shapes=["3x3"],
        pickup_configs=[PickupType.HUMBUCKER],
        spectral_centroid_range=(1500, 3000),
        typical_fundamental_range=(82, 1200),
        coaching_notes=[
            "The Les Paul's humbuckers are thick and warm. "
            "For clarity in a dense mix, a slight 2kHz presence boost "
            "on the preamp will help it cut through.",
        ],
    ),
    InstrumentModelProfile(
        name="Fender Precision Bass",
        brand="Fender", model="Precision Bass",
        family=InstrumentFamily.BASS_ELECTRIC,
        body_shapes=["double_cutaway", "contoured"],
        headstock_shapes=["inline_4"],
        pickup_configs=[PickupType.SINGLE_COIL],
        spectral_centroid_range=(400, 1500),
        typical_fundamental_range=(41, 400),
        coaching_notes=[
            "The P-Bass split-coil delivers a strong fundamental. "
            "Let it own the 60-120Hz range and high-pass the kick drum above it.",
        ],
    ),
    InstrumentModelProfile(
        name="Fender Jazz Bass",
        brand="Fender", model="Jazz Bass",
        family=InstrumentFamily.BASS_ELECTRIC,
        body_shapes=["offset", "contoured"],
        headstock_shapes=["inline_4"],
        pickup_configs=[PickupType.SINGLE_COIL],
        spectral_centroid_range=(500, 2000),
        typical_fundamental_range=(41, 400),
        coaching_notes=[
            "The Jazz Bass has more growl and midrange presence than a P-Bass. "
            "Both pickups blended gives the classic scooped tone — "
            "favor the bridge pickup for more bite in a rock mix.",
        ],
    ),
    InstrumentModelProfile(
        name="Martin D-28",
        brand="Martin", model="D-28",
        family=InstrumentFamily.GUITAR_ACOUSTIC,
        body_shapes=["dreadnought"],
        headstock_shapes=["3x3"],
        pickup_configs=[],
        spectral_centroid_range=(1800, 3500),
        typical_fundamental_range=(82, 1200),
        coaching_notes=[
            "The dreadnought body produces powerful low-end projection. "
            "Mic placement at the 12th fret avoids the boomy soundhole buildup "
            "while capturing the full frequency range.",
        ],
    ),
    InstrumentModelProfile(
        name="Shure SM58",
        brand="Shure", model="SM58",
        family=InstrumentFamily.VOCAL_MALE,  # commonly used for vocals
        body_shapes=[],
        headstock_shapes=[],
        pickup_configs=[],
        spectral_centroid_range=(1000, 4000),
        typical_fundamental_range=(80, 1200),
        coaching_notes=[
            "The SM58 has a built-in presence peak around 5kHz that helps vocals cut. "
            "Work the proximity effect: 2 inches for intimate warmth, "
            "6 inches for a natural, flat response.",
        ],
    ),
    InstrumentModelProfile(
        name="Neumann U87",
        brand="Neumann", model="U87",
        family=InstrumentFamily.VOCAL_FEMALE,
        body_shapes=[],
        headstock_shapes=[],
        pickup_configs=[],
        spectral_centroid_range=(1500, 6000),
        typical_fundamental_range=(80, 2000),
        coaching_notes=[
            "The U87 captures everything — including room reflections. "
            "Use the figure-8 or cardioid pattern wisely, and ensure "
            "there are no hard surfaces directly behind the vocalist.",
        ],
    ),
]


# ─── Visual → Audio Compatibility Matrix ─────────────────────────────────────

_VISUAL_TO_AUDIO_MAP: dict[VisualCategory, list[InstrumentFamily]] = {
    VisualCategory.GUITAR_SOLID_BODY: [InstrumentFamily.GUITAR_ELECTRIC],
    VisualCategory.GUITAR_HOLLOW_BODY: [InstrumentFamily.GUITAR_ELECTRIC, InstrumentFamily.GUITAR_ACOUSTIC],
    VisualCategory.GUITAR_ACOUSTIC: [InstrumentFamily.GUITAR_ACOUSTIC],
    VisualCategory.BASS_ELECTRIC: [InstrumentFamily.BASS_ELECTRIC],
    VisualCategory.BASS_ACOUSTIC: [InstrumentFamily.BASS_ACOUSTIC],
    VisualCategory.DRUM_KIT: [
        InstrumentFamily.DRUMS_KICK, InstrumentFamily.DRUMS_SNARE,
        InstrumentFamily.DRUMS_HIHAT, InstrumentFamily.DRUMS_CYMBAL,
        InstrumentFamily.DRUMS_TOM,
    ],
    VisualCategory.DRUM_SINGLE: [
        InstrumentFamily.DRUMS_SNARE, InstrumentFamily.DRUMS_TOM,
    ],
    VisualCategory.KEYBOARD: [
        InstrumentFamily.KEYS_PIANO, InstrumentFamily.KEYS_SYNTH,
        InstrumentFamily.KEYS_ORGAN,
    ],
}


# ─── Multimodal Fusion Engine ────────────────────────────────────────────────

class MultimodalFusion:
    """
    Cross-validates acoustic instrument detection with computer vision
    to achieve model-level identification.

    Fusion strategy:
    1. Audio classifier produces InstrumentDetection (family + pickup)
    2. Vision provides VisionDetection (shape + brand/model text + color)
    3. Cross-validate: if vision category matches audio family → boost confidence
    4. Model matching: compare vision brand/shape against INSTRUMENT_MODEL_DB
    5. Generate contextualized coaching hints
    """

    def __init__(self, sample_rate: int = 48_000):
        self._classifier = InstrumentClassifier(sample_rate)
        self._model_db = INSTRUMENT_MODEL_DB

    def fuse(
        self,
        audio: np.ndarray,
        vision_detections: list[VisionDetection] | None = None,
        environment: EnvironmentDetection | None = None,
    ) -> FusedDetection:
        # 1. Acoustic classification
        audio_det = self._classifier.classify(audio)

        if not vision_detections:
            return FusedDetection(
                instrument=audio_det,
                vision=None,
                environment=environment,
                fused_confidence=audio_det.confidence,
                model_identification=f"{audio_det.family.value} (audio-only)",
                coaching_context=audio_det.coaching_hints,
            )

        # 2. Find best-matching vision detection
        best_vision = self._find_best_vision_match(audio_det, vision_detections)

        # 3. Cross-validate & boost confidence
        fused_conf = audio_det.confidence
        model_id = f"{audio_det.family.value}"
        coaching = list(audio_det.coaching_hints)

        if best_vision:
            compatible_families = _VISUAL_TO_AUDIO_MAP.get(best_vision.category, [])
            if audio_det.family in compatible_families:
                # Audio and vision agree — boost confidence
                fused_conf = min(1.0, audio_det.confidence + best_vision.confidence * 0.3)
            else:
                # Disagreement — reduce confidence but trust vision for model text
                fused_conf = max(0.3, audio_det.confidence * 0.7)

            # 4. Model database lookup
            matched_model = self._match_model(audio_det, best_vision)
            if matched_model:
                model_id = matched_model.name
                if audio_det.pickup_type != PickupType.UNKNOWN:
                    model_id += f" ({audio_det.pickup_type.value})"
                coaching.extend(matched_model.coaching_notes)
                fused_conf = min(1.0, fused_conf + 0.15)
            elif best_vision.brand_text or best_vision.model_text:
                model_id = f"{best_vision.brand_text} {best_vision.model_text}".strip()
                if audio_det.pickup_type != PickupType.UNKNOWN:
                    model_id += f" ({audio_det.pickup_type.value})"

        # 5. Mic detection
        mic_vision = next(
            (v for v in vision_detections
             if v.category in (
                 VisualCategory.MICROPHONE_DYNAMIC,
                 VisualCategory.MICROPHONE_CONDENSER,
                 VisualCategory.MICROPHONE_RIBBON,
             )),
            None,
        )
        mic_type = ""
        if mic_vision:
            mic_type = f"{mic_vision.brand_text} {mic_vision.model_text}".strip() or mic_vision.category.value

        # 6. Speaker positions (for sweet spot engine)
        speaker_positions = []
        for v in vision_detections:
            if v.category == VisualCategory.MONITOR_SPEAKER:
                # Estimate position from bounding box center (normalized)
                cx = v.bounding_box.x + v.bounding_box.width / 2
                # Map to approximate 3D position (assumes 3m wide room)
                speaker_positions.append((
                    (cx - 0.5) * 3.0,  # x: left-right
                    1.0,                # y: ~1m height
                    -2.0,               # z: ~2m away
                ))

        # 7. Environment-specific coaching
        if environment:
            for surface in environment.reflective_surfaces:
                if "glass" in surface.lower():
                    coaching.append(
                        "Hard reflective surface detected (glass). "
                        "Rotate your setup so the mic's rejection pattern faces the glass, "
                        "or drape a heavy blanket over it."
                    )
                if "bare_wall" in surface.lower():
                    coaching.append(
                        "Untreated parallel walls detected — this can cause flutter echo. "
                        "Even a bookshelf or thick curtain on one wall will break up the problem."
                    )

        return FusedDetection(
            instrument=audio_det,
            vision=best_vision,
            environment=environment,
            fused_confidence=fused_conf,
            model_identification=model_id,
            mic_type=mic_type,
            speaker_positions=speaker_positions,
            coaching_context=coaching,
        )

    def _find_best_vision_match(
        self,
        audio_det: InstrumentDetection,
        vision_detections: list[VisionDetection],
    ) -> VisionDetection | None:
        """Find the vision detection most compatible with the audio classification."""
        best: VisionDetection | None = None
        best_score = -1.0

        for v in vision_detections:
            score = v.confidence
            compatible = _VISUAL_TO_AUDIO_MAP.get(v.category, [])
            if audio_det.family in compatible:
                score *= 2.0  # Double weight for compatible detections
            if v.brand_text:
                score *= 1.5  # Brand text found by OCR
            if score > best_score:
                best_score = score
                best = v

        return best

    def _match_model(
        self,
        audio_det: InstrumentDetection,
        vision: VisionDetection,
    ) -> InstrumentModelProfile | None:
        """Match against the instrument model database."""
        candidates: list[tuple[InstrumentModelProfile, float]] = []

        for model in self._model_db:
            score = 0.0

            # Family match
            if model.family == audio_det.family:
                score += 0.3
            elif model.family in _VISUAL_TO_AUDIO_MAP.get(vision.category, []):
                score += 0.1

            # Brand match (OCR)
            if vision.brand_text and vision.brand_text.lower() in model.brand.lower():
                score += 0.3

            # Model text match
            if vision.model_text and vision.model_text.lower() in model.model.lower():
                score += 0.3

            # Body shape match
            if vision.body_shape and vision.body_shape in model.body_shapes:
                score += 0.2

            # Headstock match
            if vision.headstock_shape and vision.headstock_shape in model.headstock_shapes:
                score += 0.15

            # Pickup type match
            if audio_det.pickup_type in model.pickup_configs:
                score += 0.15

            # Spectral centroid in expected range
            if audio_det.spectral_fingerprint:
                centroid = audio_det.spectral_fingerprint.spectral_centroid_hz
                lo, hi = model.spectral_centroid_range
                if lo <= centroid <= hi:
                    score += 0.1

            if score > 0.3:
                candidates.append((model, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
