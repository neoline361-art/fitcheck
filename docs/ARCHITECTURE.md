# Architecture

FitCheck is a Python library organized into independent modules. Core logic uses pure functions; backends and renderers are small focused classes where state is needed.

```
fitcheck/
├── __init__.py    # Public API: check(), report(), detect_drift(), detect_seasonality(), get_backend(), get_renderer(), log_to_mlflow(), log_to_dvc()
├── _version.py    # Single source of version
├── __main__.py    # python -m fitcheck
├── py.typed       # PEP 561 marker
├── check.py       # Dataset health engine
├── report.py      # Model evaluation (classification + regression)
├── drift.py       # Distribution drift detection (KS, PSI, Wasserstein, JS, Chi2, schema)
├── extensions.py  # Plugin runner + time-series validation, gap and seasonality checks
├── plugins.py     # Lightweight plugin registry and loader
├── magic.py       # Jupyter magics (%fitcheck, %%fitcheck)
├── demo.py        # Built-in demo (fitcheck demo)
├── doctor.py      # Environment health diagnosis
├── fix.py         # Transparent fix script generation
├── html.py        # Dark-mode HTML report rendering
├── fingerprint.py # SHA-256 dataset hashing, HMAC-SHA256 signing, report verification
├── cli.py         # Terminal interface (check, verify, report, drift, full, demo, doctor)
├── backends/      # Data backends: base, pandas, polars, duckdb
├── viz/           # Chart renderers: plotly interactive
└── integrations/  # Optional MLflow / DVC callbacks
tests/
examples/
benchmarks/
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure functions for core logic | Testable, composable, predictable |
| Separate modules by concern | Easy to reason about, easy to test |
| HTML reports self-contained | No external CDN, shareable offline |
| Fix scripts instead of mutation | User always controls data changes |
| No config files | Zero-boilerplate philosophy |
| Optional heavy deps are extras | polars, plotly, shap, mlflow, dvc stay out of the core install |
| Tamper-evident reports | Every HTML report embeds a visible SHA-256 fingerprint; HMAC signing optional |
| Verifiable outputs | `fitcheck verify` proves a report was not tampered with after generation |
