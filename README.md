<p align="center">
  <h1 align="center">FitCheck</h1>
  <p align="center">Zero-boilerplate ML data validation and model evaluation.</p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache 2.0">
    <img src="https://img.shields.io/badge/Tests-22%2B-brightgreen" alt="22+ tests">
  </p>
</p>

---

## Philosophy

- **Zero Config** — Pass a file path. Get answers. No YAML, no setup.
- **Immutability** — FitCheck diagnoses, never silently modifies. Fix scripts are transparent and inspectable.
- **Shareability** — Every check generates a self-contained HTML report ready for Slack, email, or GitHub.

## Installation

```bash
pip install fitcheck
```

Or install from source for development:

```bash
git clone https://github.com/neoline361-art/fitcheck.git
cd fitcheck
pip install -e ".[dev]"
```

## Quick Start

### 1. Validate a Dataset

```python
import fitcheck

# One-line health check
issues = fitcheck.check("data.csv", target="label")
# Generates: fitcheck_report.html
```

Or via CLI:

```bash
fitcheck check data.csv --target label --auto-fix
```

### 2. Evaluate a Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

metrics = fitcheck.report(model, X_test, y_test)
# Auto-detects classification vs regression
# Generates: model_report.html with metrics + plots
```

### 3. Detect Drift

```python
results = fitcheck.detect_drift("train.csv", "production.csv")
# KS test for numeric, Chi-squared for categorical
# Generates: drift_report.html
```

## What's New in v2.0

- **Auto-Fix Scripts** — `auto_fix=True` generates a transparent Python script with every fix step commented. Review before running.
- **Type Safety** — Full type hints, mypy strict mode clean.
- **CI Integration** — GitHub Actions workflow + PR bot that validates data file changes automatically.
- **22+ Tests** — pytest with 80%+ coverage across edge cases.

## Architecture

| Module | Purpose |
|--------|---------|
| `check.py` | Dataset health: missing values, duplicates, outliers, class imbalance, constant columns |
| `report.py` | Model evaluation: auto-detects classification/regression, metrics + visualizations |
| `drift.py` | Drift detection: KS test (numeric), Chi-squared (categorical) |
| `fix.py` | Transparent fix script generation from diagnostic output |
| `html.py` | Dark-mode HTML report rendering (self-contained, no external assets) |
| `cli.py` | Terminal interface: `fitcheck check`, `fitcheck report`, `fitcheck drift` |

## Development

```bash
# Run tests with coverage
pytest --cov=fitcheck --cov-report=term-missing

# Lint
ruff check fitcheck/

# Type check
mypy fitcheck/

# Demo (generates 3 HTML reports)
python demo.py
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
