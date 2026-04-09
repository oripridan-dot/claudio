# Claudio — System State

Last updated: 2026-04-09

---

## Current Status

### Core Pipeline — Intent Capture → Regeneration

| Component | Status | Notes |
|---|---|---|
| Intent Encoder (`intent/intent_encoder.py`) | ✅ Production | YIN F0 + MFCC + loudness + onset + vibrato at 250Hz, NaN/Inf sanitized |
| Intent Protocol (`intent/intent_protocol.py`) | ✅ Production | Binary wire format, delta compression (~55% savings), rms_energy serialized |
| Intent Decoder (`intent/intent_decoder.py`) | ✅ Production | 40-harmonic additive synthesis, gain smoothing, MFCC carry-forward for delta |
| Intent Stream (`intent/intent_protocol.py`) | ✅ Production | Stateful packer with FULL_FRAME, DELTA, SILENCE, KEY_FRAME modes |
| DDSP Forge (`forge/`) | ✅ Training | ForgeModel (FeatureExtractor → GRUEncoder → DDSPDecoder), calibrating on multitracks |

### Collaboration Infrastructure

| Component | Status | Notes |
|---|---|---|
| Session Manager (`collab/session_manager.py`) | ✅ Production | Room lifecycle, peer join/leave, token-bucket rate limiting (64KB/s), stale room TTL cleanup |
| Collaboration Server (`server/claudio_server.py`) | ✅ Production | WS binary + JSON signaling, JSON parse safety, snapshot-safe broadcast |
| Frontend IntentEngine (`frontend/src/engine/IntentEngine.ts`) | ✅ Production | 120Hz capture, WS reconnection with backoff, ping/pong latency measurement |
| CollabPage (`frontend/src/pages/CollabPage.tsx`) | ✅ Built | Room creation/joining UI |
| IntentVisualizer (`frontend/src/components/IntentVisualizer.ts`) | ✅ Built | Real-time intent packet visualization |

### Test Suite

| Suite | Tests | File |
|---|---|---|
| Intent Pipeline (core) | 20 | `test_intent_pipeline.py` |
| Intent Hardening (edge cases) | 15 | `test_intent_hardening.py` |
| Collab Session (room mgmt) | 9 | `test_collab_session.py` |
| Collab E2E (network/wire) | 19 | `test_collab_e2e.py` |
| DDSP Forge Model | 4 | `test_forge_model.py` |
| HRTF, Metering, Intelligence | 101 | Various |
| **Total** | **168 passed** | 1 xfailed |

### Feature Modules (Secondary Priority)

| Component | Status | Notes |
|---|---|---|
| C++ DSP Engine (`libclaudio_spatial.a`) | ✅ Builds | CMake Release, mutex-free spatial path verified |
| HRTF Engine (`hrtf_engine.py`) | ✅ Complete | 192kHz binaural rendering — feature module, not core |
| SOFA Loader | ✅ Complete | AES69-2022 HRTF loader with h5py backend + procedural fallback |
| Intelligence subsystem (`intelligence/`) | ✅ 18 modules | CLAP/PANNs backends, Gemini coach, room scanner |
| Mentor subsystem (`mentor/`) | ✅ 4 modules | Knowledge base, 15 mentor tips, progressive roadmap engine |
| Watcher subsystem (`watcher/`) | ✅ 2 modules | AI_THOUGHT_LOG pattern |

### Infrastructure

| Component | Status | Notes |
|---|---|---|
| Python package (`src/claudio/`) | ✅ Clean | Zero lint errors, all imports package-relative |
| Frontend (`frontend/`) | ✅ Dependencies synced | npm install verified |
| Studio (`studio/`) | ✅ Dependencies synced | npm install verified |
| CI pipeline | ✅ Configured | C++ build + mutex audit + Python lint/test + pip-audit |

## Version

- `pyproject.toml`: 1.1.0
- `claudio_server.py`: 1.1.0
- `/health` endpoint: 1.1.0

## Architecture Compliance

| Rule | Status | Details |
|---|---|---|
| 500-line file limit | ✅ Compliant | 0 files >500 lines |
| No inline `# noqa` | ✅ Compliant | All suppressions in `pyproject.toml` per-file-ignores |
| No mutex on audio path | ✅ Verified | CI grep audit enforced |
| Single Responsibility | ✅ Enforced | All intent/collab modules <500 lines |
| Branch protection | ✅ Enforced | Direct pushes to main blocked by CI |

## Wire Protocol Specification

| Packet Type | Size | Contents |
|---|---|---|
| SILENCE | 9 bytes | Header only (seq + ts + flags) |
| DELTA | 34 bytes | Header + F0/loudness/centroid/onset (no MFCCs) |
| FULL_FRAME | 98 bytes | Header + F0/loudness/centroid/onset + 13 MFCCs + vibrato + rms_energy |
| KEY_FRAME | 98 bytes | Periodic full frame for receiver sync |

## Latency Budget

| Stage | Time |
|---|---|
| Frontend capture (120Hz) | 8.3ms |
| Audio buffer (2048/44100) | 46.4ms |
| Network (LAN) | ~1ms |
| Server broadcast | ~0.1ms |
| Receiver regeneration | ~0.5ms |
| **Total LAN** | **~56ms** |
| **Total WAN** | **~96ms** |

## DDSP Calibration Data

| File | Channels | Sample Rate | Duration | Instrument |
|---|---|---|---|---|
| AGuitar_E.wav | 1ch | 44.1kHz | 175.9s | Acoustic Guitar |
| Bass_E.wav | 1ch | 44.1kHz | 175.9s | Bass |
| EGuitar_E.wav | 1ch | 44.1kHz | 175.9s | Electric Guitar |
| Piano_E.wav | 2ch | 44.1kHz | 175.9s | Piano |
| VocalGuide.wav | 1ch | 44.1kHz | 175.9s | Vocal |

## Recent Changes (v1.2.0-dev)

### Intent Pipeline Industrialization (4 audit rounds, 29 fixes)

- **Encoder**: NaN/Inf sanitization, vectorized MFCC filterbank (80× speedup), A-weighting activated
- **Decoder**: Vectorized inverse DCT (3.8×), gain smoothing, MFCC carry-forward for delta, silence short-circuit (85×)
- **Protocol**: Delta compression (~55% bandwidth savings), rms_energy serialized, backward-compatible wire format
- **Server**: JSON parse safety, snapshot-safe broadcast (both binary + JSON), stale room TTL cleanup
- **Session Manager**: Token-bucket rate limiting (64KB/s per sender), room lifecycle cleanup
- **Frontend**: 120Hz capture rate, WebSocket reconnection with exponential backoff, ping/pong latency measurement
- **Tests**: 33 → 67 intent/collab tests, 168 total suite passing

## Pending

- ⚠️ Complete DDSP training calibration on full multitrack dataset
- ⚠️ Build WebRTC data channels for P2P transport (replace WebSocket relay)
- ⚠️ End-to-end browser demo: mic → intent → network → regen → playback
- ⚠️ Add DDSP-based decoder option to IntentDecoder (currently additive synthesis)
