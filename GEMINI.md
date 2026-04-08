# Claudio — Sovereign Constitution

**Governed by**: TooLoo Ecosystem Governance — all rules inherit from the TooLoo `GEMINI.md` root constitution.

---

## Rule 0.A — Solo Operator Sovereignty

Claudio is built and maintained by a **solo non-coding operator** using AI agents. Every architectural decision, every code change, and every deployment must pass the solo-maintainability test:

> *"Can the operator maintain and debug this alone, with AI assistance, six months from now?"*

If not — simplify. Complexity that requires a team is **constitutional debt**.

---

## Rule 0.B — Brutal Honesty

All AI agents working on Claudio must be brutally honest. If a command fails, say so. If a module is missing, say it is missing. If an API call fails, report the actual error. Masking failure is a **constitutional violation**.

### Enforcement Clauses

- **Output Efficiency**: Go straight to the point. No filler.
- **Read Before Modifying**: Understand existing code before suggesting changes.
- **No Unnecessary Additions**: Don't add features beyond what was asked.
- **No Premature Abstractions**: Don't create helpers for one-time operations.
- **Security First**: No command injection, XSS, SQL injection, or OWASP Top 10 vulnerabilities.
- **Executing Actions with Care**: Consider reversibility and blast radius. Never bypass safety checks.

---

## Rule 1 — Additive Development

Every step of development must be additive: leverage and build upon what already exists rather than replacing, rewriting, or ignoring it.

- **Audit before adding** — check what exists before writing new code.
- **Extend, don't replace** — if something covers 70% of the need, extend it.
- **No ghost code** — unreachable code must be deleted and the deletion logged.
- **Incremental validation** — each step validated before the next begins.

---

## Branching Rules (Non-Negotiable)

All development work MUST happen on feature branches. Direct pushes to `main` are **blocked by CI**.

| Rule | Detail |
|------|--------|
| Branch naming | `feat/<name>`, `fix/<name>`, `refactor/<name>`, `docs/<name>` |
| Merge method | Pull Request only — CI must pass before merge |
| Commit to main | NEVER directly — only via merged PR |
| Branch lifecycle | Delete after merge — no stale branches |

### Before Any Major Change
> *"Create a new Git branch called `feat/<name>`. All work happens there."*

If the work fails → delete the branch. The stable app on `main` is untouched.

---

## Stack Constraints

| Layer | Technology | Rule |
|-------|-----------|------|
| Audio DSP | C++ with CMake | Lock-free rendering path — no mutex on spatial audio |
| Backend | Python + FastAPI | Package-relative imports only |
| AI Processing | Observation path only | NEVER touch the live audio path |
| Package Manager | uv | NEVER use pip, conda, or poetry |
| Tests | pytest | Full suite must pass on `main` at all times |
| Lint | ruff | Zero errors — no inline `# noqa` suppressions |

---

## Commit Message Format

```
type(scope): short description — what changed and why

Types: feat | fix | refactor | test | docs | perf | chore
```

---

## What Requires Human Approval Before AI Executes

- Any change to this file (`GEMINI.md`)
- Any change to `pyproject.toml` or dependency files
- Any `git push` to default branch
- Any new external dependency
- Any change to the CI pipeline
- Any deletion of source files

---

## Governance Documents

This repo maintains the 3-document hierarchy mandated by the Solo AI Governance KI:

| Document | Purpose |
|----------|---------|
| `1_INTENT.md` | The core product promise — intent capture → regeneration |
| `2_ARCHITECTURE_RULES.md` | Binding constraints — 500-line limit, no mutex, package imports |
| `3_SYSTEM_STATE.md` | Current system state — what works, what's broken, what's next |

*This Constitution is a living document. All updates must pass Rule 0.A (Solo Operator Sovereignty) and Rule 0.B (Brutal Honesty).*
