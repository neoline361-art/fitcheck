# Branch Protection Rules

This document lists the exact GitHub branch protection settings required for `main`.

## Required Settings

Navigate to **Settings → Branches → Branch protection rules → Edit** for `main`.

### Protect matching branches

| Setting | Value |
|---------|-------|
| Branch name pattern | `main` |

### Branch protection rules

| Setting | Value | Reason |
|---------|-------|--------|
| ✅ Require a pull request before merging | Yes | All changes go through PR review |
| ✅ Require approvals | 1 minimum | At least one human review |
| ✅ Require status checks before merging | Yes | CI must pass |
| ✅ Status checks required | `CI / quality (3.10)`, `CI / demo` | Core gate checks |
| ✅ Require branches to be up to date before merging | Yes | Prevent merge conflicts |
| ✅ Require conversation resolution | Yes | All review comments resolved |
| ✅ Require linear history | Yes (squash merge) | Clean git history |
| ❌ Require signed commits | No (optional) | Can enable later |
| ✅ Include administrators | Yes | Even repo owner must follow rules |
| ✅ Restrict who can push to matching branches | **No direct pushes** | Only via PR merge |
| ❌ Allow force pushes | No | Protects history |
| ❌ Allow deletions | No | Protects main |

### How to set up (exact steps)

1. Go to `https://github.com/neoline361-art/fitcheck/settings/branches`
2. Click **Add branch protection rule**
3. Branch name pattern: `main`
4. Enable all checkboxes listed above
5. Click **Create** / **Save changes**

### GitHub Actions required status checks

The CI workflow (`ci.yml`) runs these jobs. Add them as required checks:

```
CI / quality (3.10)
CI / quality (3.11)
CI / quality (3.12)
CI / quality (3.13)
CI / demo
```

At minimum, require `CI / quality (3.10)` and `CI / demo`.

### Verification

After enabling branch protection, verify by:

```bash
# Try to push directly (should be rejected)
git push origin main
# Expected: remote: error: GH006: Protected branch update failed
```
