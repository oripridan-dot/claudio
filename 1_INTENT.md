# Claudio — Product Intent (v2.0)

## What Claudio Is

Claudio is a real-time AI musician collaboration platform built on two parallel data channels:

1. **Audio Channel**: Neural audio codec (EnCodec) compresses live audio to **6-24 kbps** with near-transparent quality — vs 768+ kbps for raw PCM. This is the sound path.
2. **Intent Channel**: Extracts semantic performance intelligence (pitch, timbre, loudness, onset, articulation, vibrato) at 250 Hz frame rate. This is the intelligence path — it powers visualization, coaching, and AI assistance without touching the audio.

It enables musicians to play together across the internet with studio-quality fidelity at a fraction of the bandwidth of raw audio streaming.

## Who It Serves

Solo musicians, home-studio recordists, and small-band producers who want to collaborate remotely in real-time without the latency and quality penalties of traditional audio streaming.

## The Core Promise

**Any musician, anywhere, can play together as if they were in the same room.**

### Architecture (v2.0 — Hybrid EnCodec)

```
Musician → Mic → ┬─ Neural Codec Encode ───── WebRTC ───── Neural Codec Decode → Speaker
                  │  (EnCodec, 6-24 kbps)     (audio)     (near-transparent)
                  │
                  └─ Intent Extraction ─────── WebRTC ───── UI Intelligence
                     (F0, timbre, onset,       (data)      (visualization,
                      loudness, articulation)               coaching, metering)
```

### Bandwidth Comparison (proven, measured)

| Method | Bandwidth | Compression |
|--------|-----------|-------------|
| Raw PCM (16-bit, 48kHz) | 768 kbps | 1× |
| Intent packets only | ~20 kbps | 38× |
| EnCodec @ 6 kbps | ~10 kbps | 76× |
| **Hybrid (EnCodec + Intent)** | **~30 kbps** | **25×** |

## Non-Negotiable Features

| Feature | Purpose |
|---|---|
| Neural audio codec (EnCodec) | High-fidelity audio compression at 6-24 kbps with near-transparent quality |
| Intent extraction (F0, timbre, loudness, onset, articulation) | Parallel intelligence channel for AI-powered performance analysis |
| Real-time WebRTC transport | Dual-channel: audio track (codec) + data channel (intent metadata) |
| Instrument & model identification | Know exactly what is being played for UI and coaching context |
| Progressive roadmap | Guide users through setup → tracking → mixing → collaboration |
| Zero-latency observation | AI analysis runs on the observation path; the live audio path is never touched |

## Enhancement Modules (Optional Features)

| Feature | Purpose |
|---|---|
| Holographic binaural rendering (HRTF) | 192 kHz spatial audio with sub-1.5 ms head-tracking latency |
| Room acoustic scanning | RT60, room modes, flutter echo, early reflections — actionable treatment plans |
| Gesture-controlled mixing | Camera-driven fader, pan, and effect control via hand and head gestures |
| Mentor system | Real coaching advice from the perspectives of legendary engineers |
| Semantic metering | Pocket Radar, frequency collision maps, performance coaching |

## What Claudio Is Not

- A DAW. Claudio does not record, edit, or play back audio. It operates alongside the user's DAW.
- A plugin. Claudio is a standalone AI platform that communicates via WebSocket/WebRTC with its own UI.
- A replacement for human ears. Claudio is a coach — the musician makes the final call.
- A magic synthesizer. The audio channel transmits real compressed audio, not synthesized approximations.
