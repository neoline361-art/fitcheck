# Architecture

FitCheck is a pure-functional Python library organized into independent modules.

```
fitcheck/
├── __init__.py    # Public API: check(), report(), detect_drift()
├── __main__.py    # python -m fitcheck
├── check.py       # Dataset health engine
├── report.py      # Model evaluation (classification + regression)
├── drift.py       # Distribution drift detection (KS + Chi2)
├── fix.py         # Transparent fix script generation
├── html.py        # Dark-mode HTML report rendering
├── cli.py         # Terminal interface
├── .pre-commit-hooks.yaml
└── pro/           # Pro feature exports
    └── __init__.py
tests/
examples/
benchmarks/
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure functions over classes | Testable, composable, predictable |
| Separate modules by concern | Easy to reason about, easy to test |
| HTML reports self-contained | No external CDN, shareable offline |
| Fix scripts instead of mutation | User always controls data changes |
| No config files | Zero-boilerplate philosophy |
