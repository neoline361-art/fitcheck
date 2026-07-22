# Changelog

All notable changes to FitCheck are documented here.

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
