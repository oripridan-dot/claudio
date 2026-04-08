# Claudio — Architecture Rules

Binding constraints for all code in this repository. Every commit, every AI-generated change, every human edit must comply. These rules are the Critic's rulebook.

---

## 1. File Size Limit

No source file may exceed **300 lines**. If a file crosses this threshold, it must be decomposed into focused modules before merging.

**Rationale**: Files beyond 300 lines lose readability. Smaller files enforce single responsibility.

## 2. No Mutex on the Spatial Audio Path

The HRTF rendering pipeline (`hrtf_engine.py`, `dsp/spatial/`) must never use `std::mutex`, `std::lock_guard`, `std::condition_variable`, or any Python threading lock on the audio callback path. Head-tracking updates are delivered via lock-free ring buffers or atomic swaps.

**Gate**: CI runs a grep audit (`claudio-ci.yml`) that fails the build if mutex primitives appear in spatial code.

## 3. AI Observation, Not Intervention

Claudio's AI analysis runs on an **observation copy** of the audio stream. The live audio path between the musician's instrument and the DAW is never modified, delayed, or touched by AI processing.

## 4. Package-Relative Imports Only

All internal imports must use package-relative form (`from .module import X` or `from claudio.sub.module import X`). Bare module imports (`from module import X`) are prohibited — they break in monorepo and packaging contexts.

## 5. Dependency Management

- **Python**: `uv` only. No pip, no conda, no poetry.
- **C++**: CMake with explicit target linking. No global include pollution.
- **Frontend**: npm with lockfile. No yarn, no pnpm unless explicitly migrated.

## 6. Test Coverage

Every public class and every public function must have at least one test. The test suite (`tests/`) must pass with zero failures on `main` at all times. Expected failures are marked `xfail` with a documented reason.

## 7. Zero-Lint Policy

`ruff check src/ tests/` must return zero errors. Suppressions are allowed only via `pyproject.toml` `[tool.ruff.lint]` configuration with a documented rationale — never via inline `# noqa` comments.

## 8. Instrument Data Accuracy

The `INSTRUMENT_MODEL_DB` must accurately represent the physical characteristics of each instrument:
- Pickup type must match the actual hardware (e.g. P-Bass = `SPLIT_COIL`, not `SINGLE_COIL`)
- Spectral centroid ranges must reflect measured acoustic data
- Coaching notes must contain actionable, technically correct advice

## 9. Version Alignment

The version in `pyproject.toml`, `claudio_server.py` FastAPI constructor, and the `/health` endpoint must always match. A version bump in one place requires a bump in all three.

## 10. Loudness & Metering Standards

All loudness references must cite current platform-specific targets with source standards:
- Spotify: -14 LUFS integrated
- Apple Music: -16 LUFS integrated
- YouTube: -13 to -14 LUFS integrated
- Amazon Music: -14 LUFS integrated
- True peak ceiling: -1.0 dBTP (AES77-2023)
