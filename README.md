<p align="center">
  <h1 align="center">FitCheck</h1>
  <p align="center"><em>Zero-boilerplate ML data validation and model evaluation.</em></p>
  <p align="center">
    <a href="https://github.com/neoline361-art/fitcheck/actions"><img src="https://img.shields.io/github/actions/workflow/status/neoline361-art/fitcheck/ci.yml?branch=main&logo=github&label=CI" alt="CI"></a>
    <a href="https://github.com/neoline361-art/fitcheck/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache 2.0"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+"></a>
    <a href="https://github.com/neoline361-art/fitcheck/actions"><img src="https://img.shields.io/badge/Tests-30%20passing-brightgreen" alt="Tests"></a>
    <a href="https://pypi.org/project/data-fitcheck/"><img src="https://img.shields.io/pypi/v/data-fitcheck?color=blue" alt="PyPI"></a>
    <a href="https://github.com/neoline361-art/fitcheck/actions"><img src="https://img.shields.io/badge/Coverage-82%25-brightgreen" alt="Coverage"></a>
    <a href="https://github.com/neoline361-art/fitcheck/blob/main/SECURITY.md"><img src="https://img.shields.io/badge/Security-Policy-blue" alt="Security"></a>
    <a href="https://github.com/neoline361-art/fitcheck/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/Changelog-v2.0.1-blue" alt="Changelog"></a>
  </p>
</p>

---

**FitCheck** validates datasets, evaluates ML models, and detects distribution drift — all with one-line commands that generate shareable HTML reports.

| Package | Release | Stats |
|---------|---------|-------|
| data-fitcheck | `pip install data-fitcheck` | ![Platform: Linux, macOS, Windows]("https://img.shields.io/badge/platform-linux--macos--windows-lightgrey") |

## Philosophy

- **Zero Config** — Pass a file path. Get answers. No YAML, no setup.
- **Immutability** — FitCheck diagnoses, never silently modifies. Fix scripts are transparent and inspectable.
- **Shareability** — Every check generates a self-contained HTML report ready for Slack, email, or GitHub.
- **No Telemetry** — Zero outbound network calls. All computation is local.

## Installation

```bash
pip install data-fitcheck
```

From source (for development):

```bash
git clone https://github.com/neoline361-art/fitcheck.git
cd fitcheck
pip install -e ".[dev]"
```

## Quick Start — 30 Seconds

```python
import fitcheck

# 1. Validate a dataset
issues = fitcheck.check("data.csv", target="label")
# → Opens fitcheck_report.html with issues, stats, preview

# 2. Evaluate a model
metrics = fitcheck.report(model, X_test, y_test)
# → Opens model_report.html with metrics + plots

# 3. Detect drift
results = fitcheck.detect_drift("train.csv", "production.csv")
# → Opens drift_report.html with per-feature drift results
```

### CLI (same thing, no Python needed)

```bash
pip install data-fitcheck
fitcheck check data.csv --target label
fitcheck drift train.csv production.csv
fitcheck demo  # runs everything in one command
```

> 💡 **Try it right now:** `pip install data-fitcheck && fitcheck demo`

## What FitCheck Checks

| Check | Method | Severity |
|-------|--------|----------|
| Missing values | Null ratio >5% / >20% | Warning / Critical |
| Duplicate rows | `df.duplicated().sum()` | Warning |
| Constant columns | Single unique value | Warning |
| Class imbalance | Majority class >80% | Warning |
| Outliers | IQR method (1.5×) | Info |
| Numeric drift | KS test (p<0.05) | Critical |
| Categorical drift | Chi-squared (p<0.05) | Critical |

## Documentation

| Resource | Description |
|----------|-------------|
| [API Reference](docs/API.md) | Complete API documentation with parameters, returns, and examples |
| [Architecture](docs/ARCHITECTURE.md) | Module design and design decisions |
| [Design Decisions](docs/DECISIONS.md) | Why certain technical choices were made |
| [FAQ](docs/FAQ.md) | Frequently asked questions |
| [Examples](examples/basic_usage.py) | Runnable usage examples |
| [Benchmarks](benchmarks/results.md) | Performance benchmarks on standard hardware |

## Project Maturity

| Aspect | Status |
|--------|--------|
| Version | v2.0.1 — Semantic Versioning |
| Tests | 30 tests, 82% coverage |
| Type Safety | mypy strict — 11 type-arg warnings (all ndarray, no runtime impact) |
| Linting | ruff clean |
| Security | bandit + pip-audit (pickle warning expected for model loading) |
| License | Apache 2.0 |
| Platforms | Linux, macOS, Windows (Python 3.9–3.13) |
| PyPI | v2.0.1 — `pip install data-fitcheck` |

## Limitations

- Supports only pandas DataFrames (CSV, Parquet)
- Drift: KS test for numeric, Chi-squared for categorical
- Deep learning model evaluation is not implemented
- Datasets must fit in memory
- No streaming or distributed processing

## Development

```bash
# Setup
pip install -e ".[dev]"
pre-commit install

# Run all quality gates
ruff check fitcheck/
mypy fitcheck/
bandit -r fitcheck/ -x tests
pip-audit
pytest --cov=fitcheck --cov-report=term-missing

# Demo
fitcheck demo
```

## Versioning

FitCheck follows [Semantic Versioning](https://semver.org/). Given a version `MAJOR.MINOR.PATCH`:

- **MAJOR** — Breaking API changes
- **MINOR** — New features, backward compatible
- **PATCH** — Bug fixes, backward compatible

See [CHANGELOG.md](CHANGELOG.md) for per-release details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). TL;DR:
1. Tests must pass
2. Ruff + mypy must be clean
3. Add CHANGELOG entry

## Security

See [SECURITY.md](SECURITY.md). Report vulnerabilities to neoline361@gmail.com.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
