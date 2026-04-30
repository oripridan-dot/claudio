# Claudio — Sovereign Satellite

## Project Status
**Governed by TooLoo.** This project is a satellite operation of the TooLoo Factory. 

## Coding Standards (Sovereign Constitution)
- **Validity-First (Rule 0.12):** Zero placeholders. Do not output mock data arrays. Every feature must be validated via terminal execution or visual proof before being marked "Done".
- **Honest Uncertainty (Rule 7):** Do not bluff or hallucinate. Use the "Quote First" methodology before modifying files. If you do not know the answer, say "I don't know."
- **Minimize Bloat (Rule 0.5):** Do not over-engineer. No premature abstractions for one-time operations. Do not leave dead code in the repository.
- **Solo Maintainability (Rule 0.A):** Every architectural decision must be something a solo operator can easily debug 6 months from now. Keep cyclomatic complexity under 10.

## Environment Boundaries
- **Cloud Run / GCP:** Any deployment must include environment verification (e.g. `gcloud config list`) to prevent silent cloud failures.
- **Separation of Concerns:** Do not attempt to modify the `TooLoo` orchestration engine from within this repository.

## Execution Loops
- **EPCC Workflow:** Always follow Explore -> Plan -> Code -> Commit.
- **Code Hygiene:** Run the `/code-hygiene` workflow after major feature additions to remove unused imports, dead logs, and scratch files.

## Build & Run
- **Audio DSP**: C++ with CMake. Lock-free rendering path.
- **Backend**: Python + FastAPI. Package-relative imports only.
- **Package Manager**: `uv` (NEVER use pip, conda, or poetry).

## Reference
- See `GEMINI.md` for extended rules.
