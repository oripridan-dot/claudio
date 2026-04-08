# Claudio — System State

Last updated: 2026-04-08

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| C++ DSP Engine (`libclaudio_spatial.a`) | ✅ Builds | CMake Release, mutex-free spatial path verified |
| Python package (`src/claudio/`) | ✅ Clean | Zero lint errors, all imports package-relative |
| Intelligence subsystem (`intelligence/`) | ✅ 18 modules | CLAP/PANNs backends, Gemini coach, room scanner, phase/spectral analysis |
| Mentor subsystem (`mentor/`) | ✅ 4 modules | Knowledge base, 15 mentor tips, progressive roadmap engine |
| Test suite | ✅ 93 passed, 1 skipped, 1 xfailed | `test_hrtf_update_is_lock_free` skipped (needs pytest-benchmark) |
| Frontend (`frontend/`) | ✅ Dependencies synced | npm install verified |
| Studio (`studio/`) | ✅ Dependencies synced | npm install verified |
| CI pipeline | ✅ Configured | C++ build + mutex audit + Python lint/test + pip-audit |

## Version

- `pyproject.toml`: 1.1.0
- `claudio_server.py`: 1.1.0
- `/health` endpoint: 1.1.0

## Instrument Model Database

12 profiles: Fender Stratocaster, Telecaster, Precision Bass, Jazz Bass · Gibson Les Paul, SG · Rickenbacker 4003 · Taylor 814ce · Martin D-28 · Shure SM58, SM57 · Neumann U87

## Intelligence Engines

| Engine | Backend | Status |
|---|---|---|
| Instrument Classifier | CLAP (laion-clap) | ✅ Active |
| Audio Tagging | PANNs (panns-inference) | ✅ Active |
| Beat Detection | BeatNet-inspired heuristic | ✅ Active |
| Gemini Coach | Google Gemini (google-genai) | ✅ Active |
| Room Scanner | Schroeder RT60 + mode detection | ✅ Active |
| Phase Detector | Cross-correlation meter | ✅ Active |
| Sweet Spot Engine | Head-tracking compensation | ✅ Active |
| Multimodal Fusion | Audio + vision confidence merge | ✅ Active |

## Recent Changes (v1.1.0)

- Added `SPLIT_COIL` pickup type for accurate P-Bass representation
- Added 4 instrument profiles (Gibson SG, Rickenbacker 4003, Taylor 814ce, SM57)
- Updated loudness targets to multi-platform 2026 standards (AES77-2023)
- Migrated MediaPipe references from Face Mesh (468) to Face Landmarker (478)
- Fixed runtime crash in server: `scan.acoustic_advice` → `scan.treatment_plan`
- Added SOFA format references and personalisation pipeline to HRTF engine
- Hardened CI: official uv installer, pip-audit security scan
- Created governance documents (1_INTENT, 2_ARCHITECTURE_RULES, 3_SYSTEM_STATE)
- Version aligned across pyproject.toml, server, and health endpoint
- Added CLAP model selection and backend factory for neural classification
- Added Gemini-powered coaching engine with context-aware mentoring
- Added real-time intelligence loop (mic → classifier → coach)
- Added neural audio classification benchmark suite
- Added audio demo server with upload/process/compare UI
- Expanded test suite from 60 to 93 tests

## Pending

- ⚠️ Configure `pytest-benchmark` for HRTF latency tests
- ⚠️ Implement SOFA `.sofa` file loader for personalised HRTF profiles
- ⚠️ Expand `InstrumentModelDB` with keyboard and brass profiles
- ⚠️ Add `AI_THOUGHT_LOG` watcher pattern for autonomous operation
