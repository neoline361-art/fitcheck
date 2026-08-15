# Architecture

FitCheck is a pure-functional Python library organized into independent modules.

```
fitcheck/
├── __init__.py    # Public API: check(), report(), detect_drift(), detect_seasonality(), get_backend(), get_renderer(), log_to_mlflow(), log_to_dvc()
├── _version.py    # Single source of version
├── __main__.py    # python -m fitcheck
├── check.py       # Dataset health engine
├── report.py      # Model evaluation (classification + regression)
├── drift.py       # Distribution drift detection (KS, PSI, Wasserstein, JS, Chi2, schema)
├── extensions.py  # Plugin runner + time-series validation, gap and seasonality checks
├── plugins.py     # Lightweight plugin registry and loader
├── magic.py       # Jupyter magics (%fitcheck, %%fitcheck)
├── demo.py        # Built-in demo (fitcheck demo)
├── fix.py         # Transparent fix script generation
├── html.py        # Dark-mode HTML report rendering
├── cli.py         # Terminal interface
├── backends/      # Data backends: base, pandas, polars
├── viz/           # Chart renderers: base, static (matplotlib), plotly
├── integrations/  # Optional MLflow / DVC callbacks
└── pro/           # Pro feature exports
    └── __init__.py
tests/
examples/
benchmarks/
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure functions over classes | Testable, composable, predictable (renderers/backends are small, focused classes) |
| Separate modules by concern | Easy to reason about, easy to test |
| HTML reports self-contained | No external CDN, shareable offline |
| Fix scripts instead of mutation | User always controls data changes |
| No config files | Zero-boilerplate philosophy |
| Optional heavy deps are extras | polars, plotly, shap, mlflow, dvc stay out of the core install |
