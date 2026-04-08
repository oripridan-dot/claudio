---
description: Mandatory post-action cleanup — validate and remove dead code, orphan files, and debug artifacts after every change
---

# Code Hygiene Protocol — MANDATORY AFTER EVERY ACTION

## Rule: Every Change Leaves the Repo Cleaner Than It Found It

**After completing ANY code change** (feature, fix, refactor), you MUST run this cleanup checklist before committing. No exceptions.

This implements the **Broken Windows Theory** (Pragmatic Programmer) + **Janitor Agent** (Autonomous AI Ops) + **End-of-Session Cleanup** (Solo AI Governance).

---

## The Cleanup Checklist

### Step 1: Dead Code Scan
// turbo
```bash
# Python: find unused imports
ruff check . --select F401 --no-fix 2>/dev/null || true

# Python: find undefined names
ruff check . --select F821 --no-fix 2>/dev/null || true
```

Fix any findings before continuing.

### Step 2: Orphan File Detection
// turbo
```bash
# Find Python files with no imports referencing them (potential orphans)
for f in $(find src/ -name '*.py' -not -name '__init__.py' -not -name '__pycache__'); do
  basename=$(basename "$f" .py)
  refs=$(grep -rl "$basename" src/ tests/ scripts/ --include='*.py' | grep -v "$f" | wc -l)
  if [ "$refs" -eq 0 ]; then
    echo "⚠️  ORPHAN CANDIDATE: $f (0 references)"
  fi
done
```

Investigate each orphan. If truly unreferenced → delete it. If it's a standalone entry point → document why.

### Step 3: Debug Artifact Purge
// turbo
```bash
# Find debug prints, console.logs, TODO/FIXME/HACK markers
echo "=== Debug prints ==="
grep -rn 'print(' src/ --include='*.py' | grep -v 'def \|#.*print\|logging\|__name__' | head -20
echo "=== Console logs ==="
grep -rn 'console.log' frontend/ studio/ --include='*.ts' --include='*.tsx' 2>/dev/null | head -20
echo "=== TODO/FIXME/HACK ==="
grep -rn 'TODO\|FIXME\|HACK\|XXX' src/ tests/ --include='*.py' | head -20
```

- Remove all `print()` statements (use `logging` instead)
- Remove all `console.log` statements
- Address or document each TODO/FIXME with a ticket reference

### Step 4: File Size Health Check
// turbo
```bash
echo "=== Files exceeding 300 lines ==="
find src/ -name '*.py' -exec sh -c 'lines=$(wc -l < "$1"); if [ "$lines" -gt 300 ]; then echo "❌ $1: $lines lines (MUST SPLIT)"; elif [ "$lines" -gt 200 ]; then echo "⚠️  $1: $lines lines (REVIEW)"; fi' _ {} \;
```

| Lines | Status | Action |
|-------|--------|--------|
| < 150 | ✅ Healthy | None |
| 150–300 | ⚠️ Warning | Review, consider splitting |
| > 300 | ❌ Violation | MUST split into modules before committing |

### Step 5: Test Validation
// turbo
```bash
# Run the full test suite — no commit without green tests
pytest tests/ -v --tb=short 2>&1 | tail -5
```

If any test fails → fix it before committing. Never commit with broken tests.

### Step 6: Empty Directory Cleanup
// turbo
```bash
# Remove empty directories (left behind after file moves/deletes)
find . -type d -empty -not -path './.git/*' -not -path './.venv/*' -not -path './node_modules/*' -not -path './build/*' -delete -print 2>/dev/null
```

### Step 7: Stale Branch Cleanup
// turbo
```bash
# List merged branches that should be deleted
git branch --merged main | grep -v 'main\|master\|\*' | head -10
```

Delete any merged branches: `git branch -d <branch-name>`

---

## When to Run This

| Trigger | Action |
|---------|--------|
| After completing a feature | Full checklist (Steps 1-7) |
| After fixing a bug | Steps 1-5 |
| After refactoring | Full checklist (Steps 1-7) |
| Before creating a PR | Full checklist (Steps 1-7) |
| End of session | Full checklist + update `3_SYSTEM_STATE.md` |

## What Gets Committed

The cleanup itself is part of the commit. The commit message should note it:

```
feat: add semantic packetizer stub

Cleanup:
- Removed 2 orphan files
- Fixed 3 unused imports
- Purged debug prints
```

## Constitutional Enforcement

This checklist is **constitutionally mandated**. The AI agent MUST:
1. Run the cleanup after every code change
2. Report findings before committing
3. Never commit code with ❌ violations
4. Treat this as a hard gate — equivalent to tests passing
