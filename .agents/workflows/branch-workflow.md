---
description: Mandatory branching workflow — all feature work must use branches and PRs, never direct pushes to the default branch
---

# Branch Workflow — MANDATORY FOR ALL CHANGES

## Rule: No Code Lands on the Default Branch Without a PR

Every code change — feature, fix, refactor, docs — MUST go through a feature branch and Pull Request. Direct pushes to `main`/`master` are **blocked by CI** and constitute a **constitutional violation**.

This implements the **Branching for Safety** pattern (Solo AI Governance KI, §2) + **Rule 0.B.8** (Executing Actions with Care).

---

## The Workflow

### Step 1: Create a Feature Branch
// turbo
```bash
# Format: type/short-description
# Types: feat, fix, refactor, test, docs, perf, chore
git checkout -b feat/<name>
```

### Step 2: Do the Work

Make your changes. Follow `2_ARCHITECTURE_RULES.md` constraints. Keep commits atomic — one logical change per commit.

### Step 3: Run Code Hygiene
// turbo
```bash
# Run the full code-hygiene checklist before committing
# (See .agents/workflows/code-hygiene.md)
```

### Step 4: Commit with Proper Format
// turbo
```bash
# Format: type(scope): short description — what changed and why
git add -A
git commit -m "feat(scope): description"
```

Commit message rules:
- Types: `feat | fix | refactor | test | docs | perf | chore`
- Scope: the module or area changed (e.g., `buddy`, `gate_runner`, `ci`)
- Description: what changed and why — not just "updated files"

### Step 5: Push Branch and Create PR
```bash
git push origin feat/<name>
```

Then open a Pull Request on GitHub targeting the default branch (`main` or `master`).

**PR title MUST follow the same commit format**: `type(scope): description`

### Step 6: Wait for CI
The CI pipeline will run:
- Branch protection gate (blocks direct pushes)
- Quality gates (lint, tests, file size, complexity, security)
- Commit message format validation

**Do NOT merge until all checks pass.**

### Step 7: Merge and Cleanup
// turbo
```bash
# After PR is merged, clean up the local branch
git checkout main  # or master
git pull origin main
git branch -d feat/<name>
```

---

## When to Branch

| Scenario | Action |
|----------|--------|
| New feature | `feat/<name>` |
| Bug fix | `fix/<name>` |
| Code refactor | `refactor/<name>` |
| Test additions | `test/<name>` |
| Documentation updates | `docs/<name>` |
| Performance improvements | `perf/<name>` |
| Chores (deps, config) | `chore/<name>` |

## What NEVER Goes Directly to Main

- Code changes of any kind
- Configuration changes
- Dependency updates
- CI workflow changes

## Constitutional Enforcement

This workflow is **constitutionally mandated** (Rule 0.B.8 + Solo AI Governance KI §2). The AI agent MUST:
1. Create a feature branch before writing any code
2. Never commit directly to the default branch
3. Never push directly to the default branch
4. Always create a PR for review and CI validation
