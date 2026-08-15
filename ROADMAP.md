# Roadmap

## Shipped (v2.0 → v2.1)

- PyPI publishing (`pip install data-fitcheck`)
- PSI, Wasserstein, and Jensen–Shannon drift methods with automatic numeric-method selection
- Configurable thresholds via Python API and CLI
- `fitcheck full` one-command workflow (executive index report, optional model)
- CI-native CLI: `--json`, `--quiet`, `--fail-on`, exit codes 0/1/2/3
- High-cardinality, text-length, text-encoding, and time-series gap checks; schema drift detection
- Autocorrelation-based seasonality hint (`detect_seasonality`)
- Model calibration (Brier, reliability diagram), per-class error analysis, adjusted R², explained variance
- Tree feature importance with optional SHAP fallback (`pip install data-fitcheck[shap]`)
- Lightweight plugin registry + `--plugins` CLI flag
- Jupyter magics (`%fitcheck`, `%%fitcheck`)
- Pre-commit hook (repo root, corrected file pattern)
- Single source of version (`fitcheck/_version.py`)
- Reproducible builds: `requirements.lock`, Sigstore signing, CycloneDX SBOM, tag-triggered release workflow
- Optional polars loading backend (`--backend polars`) and interactive Plotly renderer (`--renderer plotly`)
- Optional MLflow / DVC integrations

## v3.1 (Next)

- [ ] Polars-native check engine (currently: polars accelerates loading, checks run on pandas)
- [ ] Mutation testing (mutmut) wired into CI with a maintained score target
- [ ] Parallel dataset validation (multi-CPU)

## Backlog

- Web UI and streaming data validation
- Deep learning model evaluation helpers (PyTorch, TensorFlow)
- GPU-accelerated drift detection
- Advanced governance (data contracts, lineage)
