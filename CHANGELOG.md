# Changelog

All notable changes to FitCheck are documented here.

## [Unreleased]

### Added
- `fitcheck full` one-command dataset, model, and optional drift workflow.
- PSI and normalized Wasserstein drift methods with automatic numeric-method selection.
- Configurable dataset thresholds through the Python API and CLI.
- Average precision, precision–recall curves, and recommended classification thresholds.
- Responsive report layouts, recommendations, embedded diagnostics, and safer HTML escaping.
- Regression tests for thresholds, drift methods, model diagnostics, and the full workflow.

### Fixed
- Pandas DataFrames are preserved during model prediction, eliminating avoidable scikit-learn feature-name warnings.
- README and API documentation now match the current 42-test project state.

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
