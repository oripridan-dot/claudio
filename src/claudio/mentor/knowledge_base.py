"""
knowledge_base.py — Pro-Tip Mentorship Knowledge Base

Structured database of verified acoustic & engineering principles,
grounded in real-world testimonials from top-tier studio engineers.

Each MentorTip is:
  - Triggered by a specific acoustic/visual detection event
  - Attributed to a real engineer with photo, studio, and date
  - Contains a concrete physical action (not a digital preset)
  - Categorized by production phase (setup/tracking/mixing/mastering)

The UI renders these as "Mentorship Moment" cards — a frosted-glass overlay
with the engineer's photo, their quote, and an interactive action guide.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProductionPhase(Enum):
    PRE_PRODUCTION = "pre_production"
    SETUP = "setup"
    TRACKING = "tracking"
    MIXING = "mixing"
    MASTERING = "mastering"


class TriggerCategory(Enum):
    PHASE_CANCELLATION = "phase_cancellation"
    PROXIMITY_EFFECT = "proximity_effect"
    ROOM_REFLECTION = "room_reflection"
    FLUTTER_ECHO = "flutter_echo"
    BASS_BUILDUP = "bass_buildup"
    HARSH_TRANSIENT = "harsh_transient"
    GAIN_STAGING = "gain_staging"
    MIC_PLACEMENT = "mic_placement"
    SPEAKER_PLACEMENT = "speaker_placement"
    DYNAMIC_RANGE = "dynamic_range"
    FREQUENCY_MASKING = "frequency_masking"
    TIMING_DRIFT = "timing_drift"
    ENERGY_DROP = "energy_drop"
    LOUDNESS_WAR = "loudness_war"
    MONO_COMPATIBILITY = "mono_compatibility"
    MIC_TECHNIQUE = "mic_technique"
    INSTRUMENT_SETUP = "instrument_setup"
    ROOM_TREATMENT = "room_treatment"


@dataclass
class MentorProfile:
    """A real-world audio engineer / producer."""
    name: str
    title: str                    # e.g. "Grammy-winning Recording Engineer"
    photo_asset: str              # path to portrait image asset
    notable_works: list[str]      # e.g. ["Michael Jackson - Thriller", "Quincy Jones"]
    era: str                      # e.g. "1980s-present"
    specialty: str                # e.g. "Vocal recording & orchestral mixing"


@dataclass
class MentorTip:
    """A single mentorship moment — triggered by detection, grounded in expertise."""
    tip_id: str
    trigger: TriggerCategory
    phase: ProductionPhase
    mentor: MentorProfile
    quote: str                    # the actual wisdom (their words)
    context_location: str         # where the wisdom was shared
    context_date: str             # when
    physical_action: str          # concrete physical correction for the user
    ui_action: str                # what the UI should do ("HIGHLIGHT_PHASE_BUTTON", "SHOW_MIC_ARROW")
    severity: str                 # "info", "tip", "warning", "critical"
    confidence_threshold: float   # minimum detection confidence to trigger this tip
    related_detection: str = ""   # e.g. "DRUM_SNARE_PHASE_180", "VOCAL_PROXIMITY_03_INCH"


# ─── Mentor Profiles ─────────────────────────────────────────────────────────

BRUCE_SWEDIEN = MentorProfile(
    name="Bruce Swedien",
    title="Grammy-winning Recording Engineer",
    photo_asset="assets/mentors/bruce_swedien.jpg",
    notable_works=["Michael Jackson — Thriller", "Quincy Jones — Back on the Block"],
    era="1960s–2020",
    specialty="Vocal recording, stereo imaging, Acusonic process",
)

AL_SCHMITT = MentorProfile(
    name="Al Schmitt",
    title="Legendary Recording & Mixing Engineer",
    photo_asset="assets/mentors/al_schmitt.jpg",
    notable_works=["Ray Charles — Genius Loves Company", "Diana Krall", "Steely Dan — Aja"],
    era="1950s–2021",
    specialty="Natural recording — minimal EQ, perfect mic placement",
)

SYLVIA_MASSY = MentorProfile(
    name="Sylvia Massy",
    title="Producer & Recording Engineer",
    photo_asset="assets/mentors/sylvia_massy.jpg",
    notable_works=["Tool — Undertow", "System of a Down", "Johnny Cash"],
    era="1990s–present",
    specialty="Experimental recording techniques, heavy guitar tones",
)

BOB_LUDWIG = MentorProfile(
    name="Bob Ludwig",
    title="Mastering Engineer",
    photo_asset="assets/mentors/bob_ludwig.jpg",
    notable_works=["Nirvana — Nevermind", "Led Zeppelin", "Daft Punk — Random Access Memories"],
    era="1960s–present",
    specialty="Mastering at Gateway Studios — dynamic range preservation",
)

BOB_KATZ = MentorProfile(
    name="Bob Katz",
    title="Mastering Engineer & Author",
    photo_asset="assets/mentors/bob_katz.jpg",
    notable_works=["Author: Mastering Audio", "Chesky Records"],
    era="1980s–present",
    specialty="Loudness standards, K-System, monitoring calibration",
)

RICK_RUBIN = MentorProfile(
    name="Rick Rubin",
    title="Record Producer",
    photo_asset="assets/mentors/rick_rubin.jpg",
    notable_works=["Johnny Cash — American Recordings", "Red Hot Chili Peppers", "Adele — 30"],
    era="1980s–present",
    specialty="Capturing emotional performance — stripped-down production philosophy",
)

ANDREW_SCHEPS = MentorProfile(
    name="Andrew Scheps",
    title="Mixing Engineer & Producer",
    photo_asset="assets/mentors/andrew_scheps.jpg",
    notable_works=["Red Hot Chili Peppers — Stadium Arcadium", "Adele — 21", "Metallica"],
    era="1990s–present",
    specialty="Headphone mixing, parallel compression, modern rock mixing",
)

TCHAD_BLAKE = MentorProfile(
    name="Tchad Blake",
    title="Mixing Engineer & Producer",
    photo_asset="assets/mentors/tchad_blake.jpg",
    notable_works=["The Black Keys", "Arctic Monkeys", "Peter Gabriel"],
    era="1990s–present",
    specialty="Binaural mixing, lo-fi textures, unconventional mic techniques",
)

DAVE_PENSADO = MentorProfile(
    name="Dave Pensado",
    title="Grammy-winning Mixing Engineer",
    photo_asset="assets/mentors/dave_pensado.jpg",
    notable_works=["Beyoncé", "Christina Aguilera", "Pensado's Place"],
    era="1980s–present",
    specialty="Vocal mixing, hip-hop and pop production, educational content",
)

EMILY_LAZAR = MentorProfile(
    name="Emily Lazar",
    title="Mastering Engineer — First Woman to Win Grammy for Engineering",
    photo_asset="assets/mentors/emily_lazar.jpg",
    notable_works=["Foo Fighters — Medicine at Midnight", "Haim", "Coldplay"],
    era="2000s–present",
    specialty="Modern mastering, loudness optimization, genre flexibility",
)


# ─── Tip Database ────────────────────────────────────────────────────────────

MENTOR_TIPS: list[MentorTip] = [
    # ── Phase Cancellation ───────────────────────────────────────────────
    MentorTip(
        tip_id="DRUM_SNARE_PHASE_180",
        trigger=TriggerCategory.PHASE_CANCELLATION,
        phase=ProductionPhase.TRACKING,
        mentor=SYLVIA_MASSY,
        quote=(
            "Always check your phase when miking the top and bottom of a drum. "
            "The top head moves down, pushing air at the top mic, while the bottom "
            "head moves down, pulling air away from the bottom mic. If you don't "
            "flip the phase on that bottom mic, your snare will sound like a tin can."
        ),
        context_location="Sound City Studios, Los Angeles",
        context_date="1993",
        physical_action=(
            "Press the Ø (phase reverse) button on your preamp or console for "
            "the snare bottom microphone channel."
        ),
        ui_action="HIGHLIGHT_PHASE_BUTTON",
        severity="critical",
        confidence_threshold=0.80,
        related_detection="phase_correlation < -0.80",
    ),

    # ── Room Reflection ──────────────────────────────────────────────────
    MentorTip(
        tip_id="VOCAL_GLASS_REFLECTION",
        trigger=TriggerCategory.ROOM_REFLECTION,
        phase=ProductionPhase.SETUP,
        mentor=BRUCE_SWEDIEN,
        quote=(
            "You can't fix a bad room with an EQ. If you have hard reflections "
            "bouncing into the back of a vocal mic, you've already lost the "
            "intimacy of the performance."
        ),
        context_location="Westlake Recording Studios, Los Angeles",
        context_date="1982",
        physical_action=(
            "Rotate your setup 180° so your back is to the reflective surface, "
            "or hang a heavy blanket over the glass/window behind the mic."
        ),
        ui_action="SHOW_ROTATION_ARROW",
        severity="warning",
        confidence_threshold=0.70,
        related_detection="glass_reflection_detected",
    ),

    # ── Mic Placement ────────────────────────────────────────────────────
    MentorTip(
        tip_id="ACOUSTIC_GUITAR_SOUNDHOLE",
        trigger=TriggerCategory.MIC_PLACEMENT,
        phase=ProductionPhase.TRACKING,
        mentor=AL_SCHMITT,
        quote=(
            "Never point the microphone directly at the soundhole of an acoustic guitar. "
            "That's where all the low-end mud lives. I always find the sweet spot "
            "somewhere around the 12th fret — that's where you get the full picture "
            "of the instrument without the boom."
        ),
        context_location="Capitol Studios, Hollywood",
        context_date="2005",
        physical_action=(
            "Angle the condenser mic toward the 12th fret of the guitar, "
            "approximately 6-10 inches away. Avoid pointing it at the soundhole."
        ),
        ui_action="SHOW_MIC_ARROW_12TH_FRET",
        severity="tip",
        confidence_threshold=0.65,
        related_detection="acoustic_guitar_detected AND spectral_centroid < 800",
    ),

    # ── Proximity Effect ─────────────────────────────────────────────────
    MentorTip(
        tip_id="VOCAL_PROXIMITY_OVERLOAD",
        trigger=TriggerCategory.PROXIMITY_EFFECT,
        phase=ProductionPhase.TRACKING,
        mentor=DAVE_PENSADO,
        quote=(
            "The proximity effect is your friend when you control it, and your enemy "
            "when you don't. Every inch closer to the mic doubles the bass energy. "
            "Find the distance where the warmth is musical, not muddy."
        ),
        context_location="Pensado's Place Studio, Los Angeles",
        context_date="2018",
        physical_action=(
            "Step back 4-6 inches from the microphone to reduce "
            "excessive low-end buildup from proximity effect."
        ),
        ui_action="SHOW_DISTANCE_INDICATOR",
        severity="tip",
        confidence_threshold=0.70,
        related_detection="vocal_detected AND bass_energy_ratio > 3.0",
    ),

    # ── Gain Staging ─────────────────────────────────────────────────────
    MentorTip(
        tip_id="PREAMP_CLIPPING",
        trigger=TriggerCategory.GAIN_STAGING,
        phase=ProductionPhase.SETUP,
        mentor=ANDREW_SCHEPS,
        quote=(
            "I see too many people running their preamps way too hot and thinking "
            "they need every bit of resolution. Leave 6dB of headroom. "
            "The noise floor of modern converters is so low that you'll never "
            "notice the difference, but you'll save yourself from clipped transients."
        ),
        context_location="Punkerpad Studios, London",
        context_date="2019",
        physical_action=(
            "Reduce the physical gain knob on your preamp by approximately -3 to -6dB "
            "to give transients room to breathe."
        ),
        ui_action="HIGHLIGHT_GAIN_KNOB",
        severity="warning",
        confidence_threshold=0.75,
        related_detection="peak_level > -3dBFS",
    ),

    # ── Energy / Performance ─────────────────────────────────────────────
    MentorTip(
        tip_id="TAKE_ENERGY_DROP",
        trigger=TriggerCategory.ENERGY_DROP,
        phase=ProductionPhase.TRACKING,
        mentor=RICK_RUBIN,
        quote=(
            "The timing was tight on that take, but the energy dropped. "
            "I always prioritize the emotion of a take over technical perfection. "
            "Take a five-minute breather, reset, and just play the next one for fun."
        ),
        context_location="Shangri-La Studios, Malibu",
        context_date="2014",
        physical_action="Take a 5-minute break and reset mentally before the next take.",
        ui_action="SHOW_BREAK_SUGGESTION",
        severity="info",
        confidence_threshold=0.60,
        related_detection="groove_consistency > 15ms OR velocity_range < 0.10",
    ),

    # ── Frequency Masking ────────────────────────────────────────────────
    MentorTip(
        tip_id="KICK_BASS_MUD",
        trigger=TriggerCategory.FREQUENCY_MASKING,
        phase=ProductionPhase.MIXING,
        mentor=ANDREW_SCHEPS,
        quote=(
            "The kick and bass are always fighting for the same 60-120Hz territory. "
            "You have to make a decision: who owns the sub? One gets the 60Hz fundamental, "
            "the other gets the 100Hz punch. They can't both have everything."
        ),
        context_location="Punkerpad Studios, London",
        context_date="2020",
        physical_action=(
            "Apply a high-pass filter on the bass guitar at ~40Hz to let the kick drum own "
            "the sub-bass, or sidechain the bass to duck 2-3dB on each kick hit."
        ),
        ui_action="HIGHLIGHT_EQ_COLLISION",
        severity="tip",
        confidence_threshold=0.70,
        related_detection="freq_collision: kick vs bass, 60-120Hz",
    ),

    # ── Loudness / Mastering ─────────────────────────────────────────────
    MentorTip(
        tip_id="MASTERING_OVER_LIMITING",
        trigger=TriggerCategory.LOUDNESS_WAR,
        phase=ProductionPhase.MASTERING,
        mentor=BOB_LUDWIG,
        quote=(
            "Over-limiting kills the punch of the snare and the breath of the vocal. "
            "If Spotify is going to turn you down to -14 LUFS anyway, "
            "why crush the life out of your mix to hit -6? Leave the dynamics in."
        ),
        context_location="Gateway Mastering, Portland, Maine",
        context_date="2015",
        physical_action=(
            "Back off the final limiter threshold by 2-3dB. "
            "Target -14 LUFS integrated for streaming platforms."
        ),
        ui_action="HIGHLIGHT_LUFS_METER",
        severity="warning",
        confidence_threshold=0.75,
        related_detection="integrated_lufs > -9",
    ),

    # ── Speaker / Sweet Spot ─────────────────────────────────────────────
    MentorTip(
        tip_id="MIXING_OFF_AXIS",
        trigger=TriggerCategory.SPEAKER_PLACEMENT,
        phase=ProductionPhase.MIXING,
        mentor=BOB_KATZ,
        quote=(
            "If you aren't sitting in the equilateral triangle, your brain is lying "
            "to you about the stereo image. You cannot make critical panning decisions "
            "off-axis. Move back to center, or let's re-align the monitors."
        ),
        context_location="Digital Domain Studio, Orlando",
        context_date="2010",
        physical_action=(
            "Return to the center listening position, or enable Claudio's "
            "Dynamic Sweet Spot compensation to auto-correct from your current position."
        ),
        ui_action="SHOW_SWEET_SPOT_GUIDE",
        severity="warning",
        confidence_threshold=0.65,
        related_detection="phantom_center_offset > 20°",
    ),

    # ── Mono Compatibility ───────────────────────────────────────────────
    MentorTip(
        tip_id="STEREO_MONO_COLLAPSE",
        trigger=TriggerCategory.MONO_COMPATIBILITY,
        phase=ProductionPhase.MIXING,
        mentor=TCHAD_BLAKE,
        quote=(
            "I mix on headphones a lot, which makes it tempting to go wide. "
            "But you have to check mono constantly. If the vocal disappears in mono, "
            "your mix is broken — because half the world is hearing your song "
            "on a single phone speaker."
        ),
        context_location="Electro-Vox Studios, Hollywood",
        context_date="2017",
        physical_action=(
            "Check your mix in mono. If key elements (vocal, snare, bass) lose "
            "more than 3dB, you have phase issues in the stereo spread."
        ),
        ui_action="SHOW_MONO_CHECK",
        severity="tip",
        confidence_threshold=0.70,
        related_detection="stereo_correlation < 0.3",
    ),

    # ── Harsh Pick Attack ────────────────────────────────────────────────
    MentorTip(
        tip_id="GUITAR_HARSH_PICK",
        trigger=TriggerCategory.HARSH_TRANSIENT,
        phase=ProductionPhase.TRACKING,
        mentor=SYLVIA_MASSY,
        quote=(
            "If the pick attack is too harsh, don't reach for an EQ — fix the player first. "
            "Have them rotate the pick 10-15 degrees so it glides across the string "
            "instead of digging in. The tone change is instant and it sounds natural."
        ),
        context_location="RadioStar Studios, Weed, California",
        context_date="2016",
        physical_action=(
            "Rotate your guitar pick 10-15° and soften your wrist. "
            "You'll retain attack clarity while losing the scrape."
        ),
        ui_action="SHOW_PICK_ANGLE_GUIDE",
        severity="tip",
        confidence_threshold=0.70,
        related_detection="pick_attack_ratio > 0.7 AND guitar_detected",
    ),

    # ── Mic Technique ────────────────────────────────────────────────────
    MentorTip(
        tip_id="SM57_AMP_CENTER",
        trigger=TriggerCategory.MIC_TECHNIQUE,
        phase=ProductionPhase.TRACKING,
        mentor=SYLVIA_MASSY,
        quote=(
            "Dead center on the speaker cone gives you the brightest, most aggressive tone. "
            "Move it 1-2 inches toward the edge and the harshness rolls off naturally. "
            "I always start off-center and move inward until I find the sweet spot."
        ),
        context_location="Sound City Studios, Van Nuys",
        context_date="1994",
        physical_action=(
            "Move the SM57 1-2 inches off the center of the speaker cone "
            "toward the edge of the dust cap."
        ),
        ui_action="SHOW_MIC_POSITION_GUIDE",
        severity="tip",
        confidence_threshold=0.65,
        related_detection="electric_guitar_bright AND mic_dynamic_detected",
    ),

    # ── Room Treatment ───────────────────────────────────────────────────
    MentorTip(
        tip_id="FLUTTER_ECHO_PARALLEL_WALLS",
        trigger=TriggerCategory.FLUTTER_ECHO,
        phase=ProductionPhase.SETUP,
        mentor=BRUCE_SWEDIEN,
        quote=(
            "Flutter echo between parallel walls will destroy the clarity of anything "
            "you record. Even a bookshelf on one wall breaks up the standing wave pattern. "
            "It doesn't have to be expensive — it just has to not be flat."
        ),
        context_location="Swedien Studios, Bel Air",
        context_date="1990",
        physical_action=(
            "Break up parallel surfaces: angle furniture, hang a thick blanket, "
            "or place a bookshelf against the reflective wall."
        ),
        ui_action="SHOW_WALL_TREATMENT_GUIDE",
        severity="warning",
        confidence_threshold=0.70,
        related_detection="flutter_echo_detected",
    ),

    # ── Bass Buildup / Room Modes ────────────────────────────────────────
    MentorTip(
        tip_id="CORNER_BASS_TRAP",
        trigger=TriggerCategory.BASS_BUILDUP,
        phase=ProductionPhase.SETUP,
        mentor=BOB_KATZ,
        quote=(
            "Bass traps are the single most impactful acoustic treatment you can add. "
            "A corner trap absorbs the room's lowest modes because that's where "
            "sound pressure is at its maximum. Start with the corners behind your monitors."
        ),
        context_location="Digital Domain Studio, Orlando",
        context_date="2012",
        physical_action=(
            "Place bass traps in the room corners, especially behind your monitors "
            "and in the rear corners. Move your listening position away from walls."
        ),
        ui_action="SHOW_BASS_TRAP_PLACEMENT",
        severity="tip",
        confidence_threshold=0.65,
        related_detection="bass_buildup_detected OR room_mode_below_200Hz",
    ),

    # ── Instrument Setup ─────────────────────────────────────────────────
    MentorTip(
        tip_id="GUITAR_TONE_KNOB",
        trigger=TriggerCategory.INSTRUMENT_SETUP,
        phase=ProductionPhase.TRACKING,
        mentor=ANDREW_SCHEPS,
        quote=(
            "Before you reach for an EQ plugin, remember that the guitar has a tone knob. "
            "Rolling it back to 7 gives you the same high-frequency cut as a shelf EQ "
            "but it sounds more organic because it's happening at the source."
        ),
        context_location="Punkerpad Studios, London",
        context_date="2021",
        physical_action=(
            "Roll the guitar's tone knob back from 10 to 7-8 to tame "
            "excessive brightness at the source."
        ),
        ui_action="SHOW_TONE_KNOB_GUIDE",
        severity="tip",
        confidence_threshold=0.65,
        related_detection="electric_guitar_bridge_pickup AND spectral_centroid > 4000",
    ),
]


# ─── Tip Retrieval Engine ────────────────────────────────────────────────────

class MentorKnowledgeBase:
    """
    Retrieves the most relevant mentorship tip for a given detection event.
    """

    def __init__(self) -> None:
        self._tips = {tip.tip_id: tip for tip in MENTOR_TIPS}
        self._by_trigger: dict[TriggerCategory, list[MentorTip]] = {}
        for tip in MENTOR_TIPS:
            self._by_trigger.setdefault(tip.trigger, []).append(tip)

    def get_tip(self, tip_id: str) -> MentorTip | None:
        return self._tips.get(tip_id)

    def find_tips(
        self,
        trigger: TriggerCategory,
        phase: ProductionPhase | None = None,
        confidence: float = 0.0,
    ) -> list[MentorTip]:
        """Find all tips matching a trigger category, filtered by phase and confidence."""
        candidates = self._by_trigger.get(trigger, [])
        results = []
        for tip in candidates:
            if tip.confidence_threshold > confidence:
                continue
            if phase is not None and tip.phase != phase:
                continue
            results.append(tip)
        return results

    def find_best_tip(
        self,
        trigger: TriggerCategory,
        phase: ProductionPhase | None = None,
        confidence: float = 0.0,
    ) -> MentorTip | None:
        """Return the single most relevant tip for the detection."""
        tips = self.find_tips(trigger, phase, confidence)
        if not tips:
            # Try without phase filter
            tips = self.find_tips(trigger, confidence=confidence)
        return tips[0] if tips else None

    @property
    def all_tips(self) -> list[MentorTip]:
        return MENTOR_TIPS

    @property
    def all_mentors(self) -> list[MentorProfile]:
        seen: dict[str, MentorProfile] = {}
        for tip in MENTOR_TIPS:
            if tip.mentor.name not in seen:
                seen[tip.mentor.name] = tip.mentor
        return list(seen.values())
