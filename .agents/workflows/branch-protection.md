---
description: Mandatory branch protection workflow — NEVER push directly to main
---

# Branch Protection Protocol — MANDATORY

## Rule: `main` is SACRED

**The `main` branch of every repository in the ecosystem is protected. Direct pushes are forbidden.**

All changes MUST go through feature branches and pull requests.

## Workflow

### 1. Create a feature branch
// turbo
```bash
git checkout -b feat/<descriptive-name>
```

### 2. Make changes on the feature branch
All code edits, file creation, and modifications happen ONLY on the feature branch.

### 3. Commit to the feature branch
// turbo
```bash
git add -A && git commit -m "<type>: <description>"
```

Commit types: `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`

### 4. Push the feature branch
// turbo
```bash
git push -u origin feat/<descriptive-name>
```

### 5. Create a Pull Request
// turbo
```bash
gh pr create --title "<type>: <description>" --body "Summary of changes" --base main
```

### 6. Merge the PR (after CI passes)
// turbo
```bash
gh pr merge --squash --delete-branch
```

### 7. Pull the updated main locally
// turbo
```bash
git checkout main && git pull origin main
```

## Protected Repositories (15 total)

| Repo | Default Branch | Protection |
|------|---------------|------------|
| `claudio` | main | ✅ PR required, admin bypass |
| `jupiter-editor` | main | ✅ PR required, admin bypass |
| `tooloo-v5` | master | ✅ PR required, admin bypass |
| `tooloo-v2` | main | ✅ PR required, admin bypass |
| `manifestation-circus` | main | ✅ PR required, admin bypass |
| `tooloo-core` | main | ✅ PR required, admin bypass |
| `tooloo-engram` | main | ✅ PR required, admin bypass |
| `tooloo-memory` | main | ✅ PR required, admin bypass |
| `tooloo-danger-room` | main | ✅ PR required, admin bypass |
| `tooloo-studio` | main | ✅ PR required, admin bypass |
| `tooloo-dna` | main | ✅ PR required, admin bypass |
| `tooloo-simulated-client` | main | ✅ PR required, admin bypass |
| `Halilit-Support-Center` | main | ✅ PR required, admin bypass |
| `Halilit-Thomann` | main | ✅ PR required, admin bypass |
| `hsc-jit-v3` | main | ✅ PR required, admin bypass |

## Admin Backdoor

If an emergency direct push to `main` is absolutely necessary:

```bash
# EMERGENCY ONLY — bypasses branch protection
git push origin main --force
```

This works because `enforce_admins` is set to `false` — the repo owner (admin) can bypass PR requirements. Regular collaborators cannot.

## Why This Exists

The USER is not a coder. They evaluate end results only. This ensures:
1. **No accidental mutations** to production code
2. **Every change is traceable** via PR history
3. **CI gates validate** before merge
4. **Clean rollback** — revert any PR if needed
