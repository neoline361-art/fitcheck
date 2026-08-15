# Contributing

## Philosophy

FitCheck follows "Diagnose, don't operate." Never mutate user data silently.

## Development Setup

```bash
git clone https://github.com/neoline361-art/fitcheck.git
cd fitcheck
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before Submitting

1. **Tests**: `pytest --cov=fitcheck` — all must pass
2. **Lint**: `ruff check fitcheck/` — clean
3. **Types**: `mypy fitcheck/` — strict, fully clean

## Code Style

- Type hints required on all public functions
- Use `from __future__ import annotations`
- Pure functions preferred over classes
- 100 char line length
- Google-style docstrings

## Commit Messages

```
<type>: <short description>

<optional body>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `security`

## Pull Request Process

1. Update CHANGELOG.md
2. Update tests for any new functionality
3. Update relevant documentation
4. Run the full gate: `ruff check fitcheck tests`, `mypy fitcheck`, `bandit -r fitcheck/ -x tests`, `pip-audit`, `pytest -W error::DeprecationWarning -W error::FutureWarning`
5. Ensure CI passes

## PR Title Convention

`type(scope): description` — e.g. `feat(check): add parquet support`
