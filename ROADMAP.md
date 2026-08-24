# Roadmap

## Shipped (v3.2.0 — 2026-08-24)

- All v2.x features (drift methods, configurable thresholds, full workflow, CI CLI, model calibration, SHAP, plugins, Jupyter magics)
- Text-encoding check, autocorrelation seasonality hint, MLflow/DVC integrations
- Polars + DuckDB loading backends (`--backend polars|duckdb`)
- Interactive Plotly renderer (`--renderer plotly`)
- `fitcheck doctor` — environment health diagnosis (`--json`, exit code 0/2)
- Edge-case hardening: empty inputs, all-NaN columns, surrogate characters, pyarrow StringDtype
- Auto-fix quality suite: verified across 5 dataset types
- Reproducible benchmark runner (`make benchmark`)
- Responsive HTML reports (≤640px, collapsible `<details>` previews)
- PEP 561 `py.typed` marker
- Makefile with test/lint/typecheck/security/audit/doctor/benchmark targets
- 142 tests at ~95% coverage, ruff clean, mypy strict clean, bandit clean

## Next

- [ ] Polars-native check engine (currently: polars accelerates loading, checks run on pandas)
- [ ] Mutation testing (mutmut) wired into CI with a maintained score target
- [ ] Parallel dataset validation (multi-CPU)

## Backlog

- Distribution plots (histograms, KDE, correlation heatmap) in HTML reports
- `fitcheck compare` command to diff two reports
- JSON/CSV output for CI pipelines beyond `--json`
- Deep learning model evaluation helpers (PyTorch, TensorFlow)
- GPU-accelerated drift detection
- Advanced governance (data contracts, lineage)
