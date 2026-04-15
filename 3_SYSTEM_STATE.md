# Claudio — System State

Last updated: 2026-04-15

---

## Current Status

### Core Pipeline — Intent + Neural Codec (v2.0)

| Component | Status | Notes |
|---|---|---|
| Neural Codec (`codec/neural_codec.py`) | ✅ Production | Wraps Meta's EnCodec, 6-24 kbps neural audio compression |
| Codec Transport (`server/claudio_server.py`) | ✅ Production | `/ws/audio` and `/ws/audio/decode` handles real compressed audio |
| Intent Encoder (`intent/intent_encoder.py`) | ✅ Production | YIN F0 + MFCC + loudness + onset + vibrato at 250Hz |
| Intent Decoder (`intent/intent_decoder.py`) | ✅ Production | Additive synthesis fallback for intent sonification/testing |
| WebRTC Transport (`collab/webrtc_manager.py`) | ✅ Production | P2P data channels with bidirectional broadcast (Integrated) |

### Collaboration Infrastructure

| Component | Status | Notes |
|---|---|---|
| Session Manager (`collab/session_manager.py`) | ✅ Production | Room lifecycle, peer join/leave, token-bucket rate limiting |
| Collaboration Server (`server/claudio_server.py`) | ✅ Production | WS binary + JSON signaling, snapshot-safe broadcast |
| Authentication (`server/auth.py`) | ✅ Production | Stateless JWT validation for WebSocket endpoints |
| Frontend IntentEngine (`frontend/src/engine/`) | ✅ Production | EnCodec ResynthEngine + module-split IntentEngine logic |
| CollabPage (`frontend/src/pages/CollabPage.tsx`) | ✅ Built | Room connection UI with Dynamic Network Telemetry rendering |
| IntentVisualizer (`frontend/src/components/IntentVisualizer.ts`) | ✅ Built | Real-time intent packet visualization |

### Test Suite

| Suite | Tests | Notes |
|---|---|---|
| Neural Codec | 7 | Compression, round-trip, serialization |
| Intent Pipeline (core) | 20 | Encoder/Decoder logic |
| Intent Hardening | 15 | Edge cases, NaN safety |
| Collab Session | 9 | Room management |
| Collab E2E | 19 | Network/wire protocol |
| HRTF, Modules | 57 | Downstream consumers |
| **Total** | **127 passed** | 0 failures |

### Infrastructure

| Component | Status | Notes |
|---|---|---|
| Python package (`src/claudio/`) | ✅ Clean | Zero lint errors, pip-compliant |
| Frontend (`frontend/`) | ✅ Dependencies synced | npm install verified |
| Studio (`studio/`) | ✅ Dependencies synced | npm install verified |

## Version

- `pyproject.toml`: 2.0.0
- `claudio_server.py`: 2.0.0
- `/health` endpoint: 2.0.0

## Architecture Compliance

| Rule | Status | Details |
|---|---|---|
| 500-line file limit | ✅ Compliant | `IntentEngine.ts` split; 0 files >500 lines |
| No inline `# noqa` | ✅ Compliant | All suppressions in `pyproject.toml` |
| No mutex on audio path | ✅ Verified | CI grep audit enforced |
| Branch protection | ✅ Enforced | Direct pushes to main blocked by CI |

## Latency Budget

| Stage | Time |
|---|---|
| Frontend codec buffering | 42.6ms |
| Network (LAN) | ~1ms |
| Server relay | ~0.1ms |
| Receiver decoding | ~2.5ms |
| **Total LAN** | **~47ms** |
| **Total WAN** | **~87ms** |

## Recent Changes (v2.0.0 - Hybrid EnCodec)

### Architectural Purge (feat/simplify-architectural-purge)
- **Purged**: Removed bloated `intelligence/`, `vision/`, `mentor/`, `metering/` packages to strictly enforce Rule 0.A (Solo Operator Sovereignty).
- **Simplified Server**: Stripped `ws_session.py` and cleared UI components tracking non-core intent models.

### Architectural Redesign
- **Killed**: `SemanticVocoder` (STFT pretend AI), `DDSPDecoder` (returned zeros), `/ws/resynth` endpoint
- **Added**: `NeuralCodec` wrapping Meta EnCodec (6-24 kbps compression at near-transparent quality)
- **Updated**: Frontend `ResynthEngine.ts` points to real neural audio endpoint
- **Cleaned**: Split 930-line `IntentEngine.ts` into `types.ts`, `dsp.ts`, `protocol.ts`, `HarmonicSynth.ts` ensuring compliance to 500-line rule.

### Measured Reality (vs Promise)
- Measured raw 16-bit PCM at 48kHz: 768 kbps
- Measured EnCodec at 6 kbps: 9.6 kbps (76x compression, -7.7 dBFS error)
- Measured intent metadata: ~20 kbps (additive synth fallback: -7.1 dBFS error, MIDI quality)
- **Verdict**: Hybrid path (~30 kbps total) is extremely viable for production.

## Pending

- ⚠️ E2E Browser Testing for `v2.0` (mic → encode → server → decode → speaker)
