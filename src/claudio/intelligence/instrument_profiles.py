"""
instrument_profiles.py — Curated Instrument Model Database

Known instrument profiles for matching against audio + vision detections.
Each profile contains physical characteristics, spectral ranges, and
actionable coaching notes grounded in real-world acoustics.

Extracted from multimodal_fusion.py for single-responsibility compliance.
"""
from __future__ import annotations

from dataclasses import dataclass

from .instrument_classifier import InstrumentFamily
from .pickup_detector import PickupType


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
        pickup_configs=[PickupType.SPLIT_COIL],
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
        name="Gibson SG",
        brand="Gibson", model="SG",
        family=InstrumentFamily.GUITAR_ELECTRIC,
        body_shapes=["double_cutaway", "thin_body"],
        headstock_shapes=["3x3"],
        pickup_configs=[PickupType.HUMBUCKER],
        spectral_centroid_range=(1600, 3200),
        typical_fundamental_range=(82, 1200),
        coaching_notes=[
            "The SG's thin mahogany body produces a focused, aggressive midrange. "
            "It cuts through dense mixes naturally — avoid over-boosting 2kHz "
            "or the upper mids will become fatiguing on long listens.",
        ],
    ),
    InstrumentModelProfile(
        name="Rickenbacker 4003",
        brand="Rickenbacker", model="4003",
        family=InstrumentFamily.BASS_ELECTRIC,
        body_shapes=["cresting_wave", "bound"],
        headstock_shapes=["inline_4"],
        pickup_configs=[PickupType.SINGLE_COIL],
        spectral_centroid_range=(600, 2200),
        typical_fundamental_range=(41, 400),
        coaching_notes=[
            "The Rickenbacker 4003 has a distinctive bright, clanky attack "
            "from its single-coil pickups and through-neck construction. "
            "Use the Rick-O-Sound stereo output to process the neck and "
            "bridge pickups independently for maximum tonal control.",
        ],
    ),
    InstrumentModelProfile(
        name="Taylor 814ce",
        brand="Taylor", model="814ce",
        family=InstrumentFamily.GUITAR_ACOUSTIC,
        body_shapes=["grand_auditorium"],
        headstock_shapes=["3x3"],
        pickup_configs=[PickupType.PIEZO],
        spectral_centroid_range=(1600, 3200),
        typical_fundamental_range=(82, 1200),
        coaching_notes=[
            "The Grand Auditorium body balances projection and clarity — "
            "less boomy than a dreadnought but still full-bodied. "
            "When using the ES2 pickup, blend it with a condenser mic "
            "at the 12th fret for the most natural captured tone.",
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
        family=InstrumentFamily.VOCAL_MALE,
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
        name="Shure SM57",
        brand="Shure", model="SM57",
        family=InstrumentFamily.GUITAR_ELECTRIC,
        body_shapes=[],
        headstock_shapes=[],
        pickup_configs=[],
        spectral_centroid_range=(1200, 5000),
        typical_fundamental_range=(80, 4000),
        coaching_notes=[
            "The SM57 is the industry standard for guitar cabs and snare drum. "
            "On a guitar cab, start 1 inch off-centre of the speaker cone — "
            "moving toward the edge rolls off the high-frequency bite, "
            "moving to the centre adds presence and aggression.",
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
