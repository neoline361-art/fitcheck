# Roadmap

## Shipped (v2.0 → v2.1)

- PyPI publishing (`pip install data-fitcheck`)
- PSI + Wasserstein drift methods with automatic numeric-method selection
- Configurable thresholds via Python API and CLI
- `fitcheck full` one-command workflow (executive index report, optional model)
- CI-native CLI: `--json`, `--quiet`, `--fail-on`, exit codes 0/1/2/3
- High-cardinality, text-length, and time-series gap checks; schema drift detection
- Jensen–Shannon drift method
- Model calibration (Brier, reliability diagram) and per-class error analysis
- Lightweight plugin registry + `--plugins` CLI flag
- Jupyter magics (`%fitcheck`, `%%fitcheck`)
- Pre-commit hook (repo root, corrected file pattern)
- Single source of version (`fitcheck/_version.py`)

## v3.0 (Next)

- [ ] Interactive Plotly reports (CDN with offline matplotlib fallback)
- [ ] Polars backend for 10M+ rows (requires a backend-agnostic check engine)
- [ ] Text encoding detection, time-series seasonality (STL decomposition)
- [ ] SHAP feature importance (optional dependency)
- [ ] Reproducible builds: lock file, Sigstore signing, CycloneDX SBOM, release workflow
- [ ] Mutation testing (mutmut) to validate test quality

## Backlog

- Parallel dataset validation (multi-CPU)
- MLflow / DVC integrations
- Web UI and streaming data validation
- Deep learning model evaluation helpers (PyTorch, TensorFlow)
- GPU-accelerated drift detection
