"""
gemini_coach.py — Gemini-Powered Real-Time Coaching Engine

Generates context-aware coaching from Gemini based on instrument detection.
Rate-limited (10s), cached, with curated fallback tips when API unavailable.
Runs on the intelligence thread — never on the audio thread.
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class CoachingContext:
    """Context sent to Gemini for coaching generation."""
    instrument: str                    # e.g. "electric guitar (humbucker)"
    confidence: float                  # classification confidence
    spectral_centroid_hz: float = 0.0  # brightness indicator
    spectral_rolloff_hz: float = 0.0   # high-freq energy cutoff
    transient_sharpness: float = 0.0   # 0=soft, 1=extremely sharp attack
    rms_level: float = 0.0            # current volume level
    session_duration_s: float = 0.0    # how long they've been playing
    recent_instruments: list[str] = field(default_factory=list)
    pickup_type: str = "unknown"
    playing_style: str = "unknown"     # "fingerpicking", "strumming", etc.


@dataclass
class CoachingResponse:
    """Response from the Gemini coaching engine."""
    tip: str                     # the main coaching message
    category: str                # "technique", "tone", "dynamics", "encouragement"
    confidence: float            # how confident the coach is in this advice
    source: str = "gemini"       # "gemini", "fallback", "cached"
    latency_ms: float = 0.0


# ─── Fallback Tips ────────────────────────────────────────────────────────────
# Used when Gemini is unavailable or rate-limited

FALLBACK_TIPS: dict[str, list[str]] = {
    "guitar_electric": [
        "Try rolling your tone knob to 7 — you'll keep the bite but lose the ice-pick.",
        "Your pick angle affects your tone more than your amp. Experiment with 15° rotation.",
        "If you're hearing fret buzz, check your relief before your action height.",
    ],
    "guitar_acoustic": [
        "Anchor your pinky on the soundboard for stability during fingerpicking passages.",
        "The sweet spot for strumming is where the neck meets the body — not over the soundhole.",
        "Try alternating between nail and pad of your thumb for tonal variety.",
    ],
    "bass_electric": [
        "Your right-hand position defines your tone: bridge = bright, neck = warm.",
        "Ghost notes between your main notes add pocket. Keep them at 30% volume.",
        "Try floating your thumb on the pickup — it mutes and anchors simultaneously.",
    ],
    "drums_snare": [
        "Your backbeat is the spine of the groove. Make it the loudest, most consistent thing.",
    ],
    "keys_piano": [
        "Curved fingers, loose wrists. Tension is the enemy of both speed and tone.",
        "Use the weight of your arm, not finger strength, for forte passages.",
        "Pedal changes should happen just AFTER the beat, not on it — prevents blurring.",
    ],
    "vocal_male": [
        "Breathe from your diaphragm, not your chest. Your belly should expand.",
        "Keep the mic 2-3 inches from your lips. Closer = more bass (proximity effect).",
    ],
}


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Claudio Coach — a world-class music production mentor embedded in the Claudio audio intelligence platform.

Your role: Provide ONE concise, actionable coaching tip based on the real-time audio analysis data you receive.

Rules:
1. Be specific and physical — tell them exactly what to do with their hands, body, or instrument
2. Never recommend software settings or plugins — only physical actions
3. Keep tips under 2 sentences
4. Be encouraging but honest — acknowledge what's working before suggesting improvement
5. Reference the specific acoustic data when relevant (e.g., "I'm hearing a lot of energy above 4kHz")
6. Rotate between technique, tone, dynamics, and encouragement categories
7. If you detect fatigue patterns (inconsistency over time), suggest a break

You will receive JSON context about the current detection. Respond with a JSON object:
{
  "tip": "Your coaching message",
  "category": "technique|tone|dynamics|encouragement"
}"""


class GeminiCoach:
    """
    Gemini-powered coaching engine with rate limiting and fallback.
    """

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        rate_limit_s: float = 10.0,
        cache_size: int = 50,
    ):
        self._model_name = model
        self._rate_limit = rate_limit_s
        self._last_call_time = 0.0
        self._client = None
        self._available = False
        self._cache: OrderedDict[str, CoachingResponse] = OrderedDict()
        self._cache_size = cache_size
        self._total_calls = 0
        self._total_cached = 0
        self._total_fallbacks = 0

        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Gemini client — fails gracefully if no API key."""
        try:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                self._client = genai.Client(api_key=api_key)
                self._available = True
                print("[GeminiCoach] Initialized with API key")
            else:
                # Try Vertex AI
                project = os.environ.get("GOOGLE_CLOUD_PROJECT")
                if project:
                    self._client = genai.Client(
                        vertexai=True,
                        project=project,
                        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                    )
                    self._available = True
                    print(f"[GeminiCoach] Initialized via Vertex AI (project: {project})")
                else:
                    print("[GeminiCoach] No API key or Vertex AI project found — using fallback tips")
        except ImportError:
            print("[GeminiCoach] google-genai not installed — using fallback tips")
        except Exception as e:
            print(f"[GeminiCoach] Init failed: {e} — using fallback tips")

    def get_coaching(self, ctx: CoachingContext) -> CoachingResponse:
        """
        Get a coaching tip for the current context.
        Rate-limited — returns cached or fallback if called too frequently.
        """
        now = time.time()

        # Rate limit check
        if now - self._last_call_time < self._rate_limit:
            # Check cache first
            cache_key = self._context_hash(ctx)
            if cache_key in self._cache:
                self._total_cached += 1
                return self._cache[cache_key]
            # Return fallback
            return self._fallback(ctx)

        # Check cache
        cache_key = self._context_hash(ctx)
        if cache_key in self._cache:
            self._total_cached += 1
            return self._cache[cache_key]

        # Try Gemini
        if self._available:
            try:
                response = self._call_gemini(ctx)
                self._last_call_time = now
                self._total_calls += 1

                # Cache the response
                self._cache[cache_key] = response
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

                return response
            except Exception as e:
                print(f"[GeminiCoach] API error: {e}")
                self._total_fallbacks += 1
                return self._fallback(ctx)

        # No API available
        self._total_fallbacks += 1
        return self._fallback(ctx)

    def _call_gemini(self, ctx: CoachingContext) -> CoachingResponse:
        """Make the actual Gemini API call."""
        import json

        from google.genai import types

        t0 = time.perf_counter()

        # Build the context message
        context_json = json.dumps({
            "instrument": ctx.instrument,
            "confidence": round(ctx.confidence, 3),
            "spectral_centroid_hz": round(ctx.spectral_centroid_hz),
            "spectral_rolloff_hz": round(ctx.spectral_rolloff_hz),
            "transient_sharpness": round(ctx.transient_sharpness, 2),
            "rms_level": round(ctx.rms_level, 4),
            "session_duration_minutes": round(ctx.session_duration_s / 60, 1),
            "pickup_type": ctx.pickup_type,
            "playing_style": ctx.playing_style,
            "recent_instruments": ctx.recent_instruments[-5:],
        })

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.7,
            max_output_tokens=200,
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=f"Current detection context:\n{context_json}",
            config=config,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Parse response
        try:
            result = json.loads(response.text)
            return CoachingResponse(
                tip=result.get("tip", "Keep playing — you're sounding great!"),
                category=result.get("category", "encouragement"),
                confidence=ctx.confidence,
                source="gemini",
                latency_ms=latency_ms,
            )
        except (json.JSONDecodeError, AttributeError):
            return CoachingResponse(
                tip=response.text[:200] if response.text else "Keep at it!",
                category="encouragement",
                confidence=ctx.confidence,
                source="gemini",
                latency_ms=latency_ms,
            )

    def _fallback(self, ctx: CoachingContext) -> CoachingResponse:
        """Return a hardcoded fallback tip based on the instrument."""
        instrument_key = ctx.instrument.split("(")[0].strip().lower().replace(" ", "_")

        # Find matching tips
        tips = FALLBACK_TIPS.get(instrument_key, [])
        if not tips:
            # Try partial match
            for key, t in FALLBACK_TIPS.items():
                if key in instrument_key or instrument_key in key:
                    tips = t
                    break

        if not tips:
            tips = [
                "Focus on consistency — play the same passage 5 times in a row with identical dynamics.",
                "Record yourself and listen back. Your ears hear differently when you're not playing.",
                "The best musicians practice slowly. Speed comes from muscle memory, not force.",
            ]

        # Rotate through tips based on time
        idx = int(time.time() / self._rate_limit) % len(tips)
        return CoachingResponse(
            tip=tips[idx],
            category="technique",
            confidence=ctx.confidence,
            source="fallback",
        )

    @staticmethod
    def _context_hash(ctx: CoachingContext) -> str:
        """Create a hash key for caching similar contexts."""
        # Round values to bucket similar contexts together
        key = f"{ctx.instrument}_{int(ctx.spectral_centroid_hz / 500)}_{int(ctx.transient_sharpness * 10)}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @property
    def stats(self) -> dict:
        return {
            "available": self._available,
            "model": self._model_name,
            "total_calls": self._total_calls,
            "total_cached": self._total_cached,
            "total_fallbacks": self._total_fallbacks,
            "cache_size": len(self._cache),
        }
