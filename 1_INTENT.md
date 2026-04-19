# Claudio — Product Intent (v4.0)

## What Claudio Is

Claudio is a real-time AI musician collaboration platform built on two parallel channels:

1. **Audio Channel**: Real Opus audio at 128 kbps (perceptually transparent, stereo, 48kHz). This is the sound path — the musician's actual performance, unmodified.
2. **Intent Channel**: Extracts semantic performance intelligence (pitch, timbre, loudness, onset, articulation, vibrato) at 120 Hz frame rate. This is the intelligence path — it powers visualization, coaching, and AI assistance without touching the audio.

It enables musicians to play together across the internet with studio-quality fidelity and an AI co-pilot that sees every note, hears every beat, and helps them sound better together.

## Who It Serves

Solo musicians, home-studio recordists, and small-band producers who want to collaborate remotely in real-time without the latency and quality penalties of traditional audio streaming — with intelligent, AI-powered performance insight that no other platform offers.

## The Core Promise

**Any musician, anywhere, can play together as if they were in the same room — with an AI co-pilot that sees every note, hears every beat, and helps them sound better together.**

### The Three Laws of Claudio Audio

1. **Microphone Principle**: Never modify the primary audio path. Observe it, transport it, deliver it.
2. **Fiber Optic Principle**: The transport layer preserves signal, not understands it. Opus at 128kbps is perceptually transparent. Use it.
3. **AI Edge Principle**: AI enhances at the edges only — suppress noise before encoding, extend bandwidth after decoding, conceal packet loss during transport. AI never replaces the signal chain.

### Architecture (v4.0 — Opus + Intent Intelligence)

```
Musician → Mic → AudioContext (48kHz, raw, no browser processing)
                    │
                    ├─── WebRTC Audio Track ─────────────────→ Remote Speaker
                    │    Opus @ 128kbps, stereo                (REAL AUDIO)
                    │    echoCancellation: false
                    │    noiseSuppression: false
                    │    autoGainControl: false
                    │
                    ├─── AnalyserNode → Intent Extraction ──→ DataChannel ──→ Remote UI
                    │    f0, loudness, melBands, onset          ~20kbps      (visualization,
                    │    120Hz frame rate (observation only)                   coaching,
                    │                                                         smart features)
                    │
                    └─── [EMERGENCY] DDSP Neural Fallback
                         Activates ONLY when: packetLoss > 25%
                         AND jitterMs > 200 AND no remote audio stream
                         Crossfades back to Opus when network recovers
```

### Bandwidth Profile

| Channel | Bandwidth | Purpose |
|---------|-----------|---------|
| Opus Audio Track | ~128 kbps | Real audio (the sound you hear) |
| Intent DataChannel | ~20 kbps | Musical intelligence (what AI sees) |
| **Total** | **~148 kbps** | Full collaboration experience |

## Non-Negotiable Features

| Feature | Purpose |
|---|---|
| Real Opus audio (128kbps, stereo, 48kHz) | Studio-grade audio — the actual performance, unmodified |
| Intent extraction (F0, timbre, loudness, onset) | Parallel intelligence channel for AI-powered performance analysis |
| Real-time WebRTC transport | Audio track (Opus) + Data channel (intent metadata) |
| Three Laws enforcement | Audio path is sacred — AI observes, never replaces |
| Emergency DDSP fallback | Neural synthesis activates only during catastrophic network failure |

## AI Intelligence Features (The Commercial Moat)

| Feature | How It Works |
|---------|-------------|
| Live Tuner | f0 extraction → real-time pitch display with cent deviation |
| Onset Sync Monitor | Compare onset timestamps across peers → show timing drift |
| Frequency Collision Map | Overlay peer spectral centroids → warn when instruments clash |
| Dynamic Auto-Panning | Use peer count + instrument type to auto-position in stereo field |
| Smart Ducking | When one peer's loudness spikes, slightly attenuate others |
| Performance Timeline | Record intent data (tiny) → replay pitch/loudness contours for review |
| AI Coach | Feed intent history to LLM → "Your timing drifted 15ms on the bridge" |

## What Claudio Is Not

- A DAW. Claudio does not record, edit, or play back audio.
- A plugin. Claudio is a standalone AI platform with its own UI.
- A replacement for human ears. Claudio is a coach — the musician makes the final call.
- A magic synthesizer. The audio channel transmits real compressed audio, not synthesized approximations. DDSP exists solely as an emergency network fallback.
