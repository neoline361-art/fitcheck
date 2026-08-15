# Changelog

All notable changes to FitCheck are documented here.

## [3.1.2] - 2026-08-15

### Fixed
- Release workflow SBOM step: `cyclonedx-py environment --outfile` is invalid in cyclonedx-bom 7.x — corrected to `-o` so the release pipeline (publish, Sigstore, SBOM, GitHub release) completes green.

## [3.1.1] - 2026-08-15

### Fixed
- Release pipeline artifact generation: Sigstore signing, CycloneDX SBOM, and GitHub release assets. v3.1.0 published to PyPI but the workflow stopped at the SBOM step; 3.1.1's run then succeeded at publish and Sigstore but hit the same SBOM failure — fully resolved in 3.1.2 with the corrected CLI flag.

## [3.1.0] - 2026-08-15

### Added
- Text-encoding check (`text_encoding`) flags object columns with non-UTF8 encodable characters.
- Autocorrelation-based seasonality hint via `fitcheck.detect_seasonality(series, period=None)`.
- Optional SHAP fallback for feature importance (`pip install data-fitcheck[shap]`).
- Optional polars loading backend: `fitcheck check data.parquet --backend polars` (`pip install data-fitcheck[polars]`).
- Optional interactive Plotly renderer: `fitcheck report ... --renderer plotly` (`pip install data-fitcheck[plotly]`); vendored Plotly JS keeps reports offline.
- Optional MLflow / DVC callbacks (`log_to_mlflow`, `log_to_dvc`).
- GitHub Action template: `.github/workflows/fitcheck-action.yml.example`.
- `fitcheck demo` supports `--no-browser` and `--output-dir`; duplicate `[1/3]` progress counter removed.

### Fixed
- Deprecation warnings: `is_categorical_dtype` replaced with `isinstance(dtype, pd.CategoricalDtype)` (pandas 3.0 readiness); Chi-squared drift no longer triggers the positional-index FutureWarning.

## [2.1.0] - 2026-08-15

### Added
- CI-native CLI: `--json`, `--quiet`, `--fail-on {info,warning,critical}`, and exit codes 0/1/2/3 (3 = runtime error).
- Multi-file dataset checks: `fitcheck check data1.csv data2.csv`.
- High-cardinality and text-length checks; time-series gap detection (`--time-column`).
- Jensen–Shannon drift method and schema drift detection (missing columns, dtype changes).
- Model calibration (Brier score, reliability diagram) and per-class error analysis; adjusted R² and explained variance for regression.
- Lightweight plugin registry (`fitcheck.plugins`) with a `--plugins` CLI flag.
- Jupyter magics `%fitcheck` / `%%fitcheck` (`pip install data-fitcheck[jupyter]`).
- `fitcheck full` works without `--model` and writes an executive index report.
- `fitcheck full` one-command dataset, model, and optional drift workflow.
- PSI and normalized Wasserstein drift methods with automatic numeric-method selection.
- Configurable dataset thresholds through the Python API and CLI.
- Average precision, precision–recall curves, and recommended classification thresholds.
- Responsive report layouts, recommendations, embedded diagnostics, and safer HTML escaping.
- Regression tests for thresholds, drift methods, model diagnostics, and the full workflow.

### Changed
- Single source of version: `fitcheck/_version.py` is read by both hatchling and the package.
- Pre-commit hook moved to the repository root with a corrected `files` pattern.
- Reproducible installs: `requirements.lock` (compiled for Python 3.10) pins runtime and dev dependencies; CI installs from the lock file.
- Tag-triggered release workflow: locked build, trusted PyPI publishing, Sigstore signing, CycloneDX SBOM, and GitHub release attachments.
- Dropped Python 3.9 support (EOL; vulnerable transitive dependencies have no patched 3.9 releases). Minimum is now Python 3.10.
- CLI runtime errors (missing file, invalid config) now exit with code 3 instead of raising.
- Removed generated demo artifacts from the package directory; test-data generator moved to `benchmarks/gen_data.py`.

### Fixed
- Pandas DataFrames are preserved during model prediction, eliminating avoidable scikit-learn feature-name warnings.
- README and API documentation now match the current project state.

## [2.0.2] - 2026-08-09

### Fixed
- mypy strict is now fully clean: bare `np.ndarray` annotations replaced with `numpy.typing.NDArray` (11 type-arg / no-any-return errors)
- ruff clean across `tests/` (removed unused imports, renamed ambiguous variables)

### Added
- CLI command tests (`tests/test_cli.py`): check, check --target/--auto-fix, report, drift, missing-file, no-command — total 36 tests, 95% coverage

## [2.0.1] - 2026-07-25

### Fixed
- `fitcheck demo` now works from any directory (moved `demo.py` into the package)
- README badge claims now match reality (honest about mypy warnings)
- 4 f-string lint errors in demo.py (ruff F541)
- Console error on missing `sev` variable in HTML report generation

### Added
- CLI smoke tests (2 tests, bringing total to 30)
- PyPI publishing: `pip install data-fitcheck`
- "Try it now" prompt in README

### Changed
- Package renamed to `data-fitcheck` for PyPI (original `fitcheck` was taken)
- README: honest about maturity, no false claims
- README: install via `pip install data-fitcheck` instead of git+https

## [2.0.0] - 2026-07-23

### Added
- Dataset health check: missing values, duplicates, constant columns, class imbalance, outliers
- Model evaluation: auto-detect classification vs regression, metrics + plots
- Drift detection: KS test (numeric), Chi-squared (categorical)
- Auto-fix: transparent Python fix scripts — never silent mutation
- Dark-mode HTML reports for all three check types
- CLI: `fitcheck check`, `fitcheck report`, `fitcheck drift`, `fitcheck demo`
- Full type hints (mypy strict mode clean)
- 28 tests with 82% coverage
- CI pipeline: ruff → mypy → pytest on Python 3.9–3.13
- PR gate: automatic data validation on CSV/Parquet changes
- Apache 2.0 license
- pyproject.toml (hatchling build)

### Changed
- Complete rewrite from v1 web app to Python CLI library

### Removed
- Web app (React/Express frontend) — replaced by Python library

## [1.0.0] - 2026-06-30

### Added
- Initial web app release (React + Express + Gemini AI)
- Basic data scanning and model evaluation UI
