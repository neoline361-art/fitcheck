<h1 align="center">FitCheck</h1>
<p align="center"><em>Zero-boilerplate ML data validation, model evaluation, and drift detection.</em></p>
<p align="center">
  <a href="https://github.com/neoline361-art/fitcheck/actions"><img src="https://img.shields.io/github/actions/workflow/status/neoline361-art/fitcheck/ci.yml?branch=main&logo=github&label=CI" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/neoline361-art/fitcheck/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache 2.0"></a>
  <a href="https://github.com/neoline361-art/fitcheck/actions"><img src="https://img.shields.io/badge/Tests-80%20passing-brightgreen" alt="Tests"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/neoline361-art/fitcheck/main/assets/screenshots/fitcheck_one_pager.png" width="85%" alt="FitCheck overview">
</p>

FitCheck is a local-first toolkit for answering three questions quickly: **Is this dataset healthy? Is this model behaving? Has production data changed?** Every workflow produces a self-contained HTML report that can be opened locally and shared in a pull request, Slack, or an incident review.

## Why FitCheck

FitCheck is intentionally opinionated. It is zero-config for the common path, never mutates data silently, makes recommendations understandable to beginners, and keeps advanced diagnostics available without making the first command complicated. It does not send telemetry or require a hosted service.

| Principle | What it means in practice |
|---|---|
| Zero friction | Pass a CSV, Parquet file, DataFrame, or trained model and receive a report. |
| Read-only by default | Checks diagnose problems; optional fix scripts are generated separately for inspection. |
| Local and private | Reports and statistical calculations run locally. |
| Progressive disclosure | The simple API remains small while full workflows expose deeper diagnostics. |
| Shareable output | Reports are standalone HTML with responsive styling and embedded plots. |

## Benchmarks

Measured on this machine (Intel Core i5-6500, 8 GB RAM, Python 3.13, Parrot OS) with `fitcheck check --quiet`; best of two wall-clock runs. Your hardware will differ.

| Dataset size | pandas backend | polars backend |
|---|---|---|
| 100k rows | 2.32 s | 1.97 s |
| 1M rows | 2.93 s | 2.60 s |

<p align="center">
  <img src="https://raw.githubusercontent.com/neoline361-art/fitcheck/main/assets/charts/fitcheck_benchmark_comparison.png" width="85%" alt="Benchmark comparison and feature scorecard">
</p>

*Evidently and Deepchecks timings in the chart are community-reported estimates, not measured here — only the FitCheck numbers are measured. The feature-scorecard values are subjective availability scores; those tools offer deeper production-monitoring features.*

FitCheck is intentionally lightweight: it targets the common daily checks (missing values, duplicates, drift, model health) in seconds, not full production monitoring. For comprehensive statistical suites and richer monitoring, [Evidently](https://github.com/evidentlyai/evidently) and [Deepchecks](https://github.com/deepchecks/deepchecks) are excellent tools — FitCheck is the fast pre-flight check you run before every training run.

Both backends produce identical diagnostics. The optional polars backend (`--backend polars`) is fastest on large files — it loads and converts to pandas for analysis, so the speedup grows with the cost of CSV/Parquet loading.

## Installation

```bash
pip install data-fitcheck
```

For development:

```bash
git clone https://github.com/neoline361-art/fitcheck.git
cd fitcheck
pip install -e ".[dev]"
```

## The one-command workflow

```bash
fitcheck full data.csv \
  --target label \
  --model model.joblib \
  --reference train.csv \
  --auto-fix \
  --output-dir fitcheck_reports
```

This creates a dataset report, model evaluation report, optional drift report, and—when issues are found—a transparent fix script. The workflow uses the dataset columns other than `label` as model features, so it is best suited to a model trained on the same feature schema.

## Python API

```python
import fitcheck

issues = fitcheck.check("data.csv", target="label")
metrics = fitcheck.report(model, X_test, y_test)
results = fitcheck.detect_drift("train.csv", "production.csv", method="auto")
```

The dataset check accepts threshold overrides without requiring a configuration file:

```python
fitcheck.check(
    "data.csv",
    target="label",
    config={
        "missing_warning": 0.05,
        "missing_critical": 0.20,
        "outlier_threshold": 0.01,
    },
)
```

## What FitCheck checks

| Area | Built-in diagnostics |
|---|---|
| Dataset health | Missing values, duplicates, constants, class imbalance, IQR outliers, high cardinality, text-length skew, text-encoding warnings |
| Time series | Timestamp parsing, monotonicity, duplicates, and frequency gaps (`--time-column`) |
| Model classification | Accuracy, precision, recall, F1, confusion matrix, ROC/AUC, average precision, precision–recall curve, recommended threshold, Brier score, calibration curve, per-class error analysis, and tree feature importance |
| Model regression | MSE, RMSE, MAE, R², adjusted R², explained variance, residual analysis, actual-versus-predicted plot, and tree feature importance |
| Drift | Automatic KS/PSI selection for numeric data, explicit Wasserstein and Jensen–Shannon distance, Chi-squared categorical comparisons, and schema drift (missing columns / dtype changes) |
| Seasonality | Autocorrelation-based hint for repeatable time-series patterns (`detect_seasonality`) |
| Reports | Severity badges, recommendations, responsive tables, embedded plots, and no external assets. Optional interactive Plotly charts (`--renderer plotly`) |
| Integrations | Optional MLflow logging and DVC metrics callbacks (`log_to_mlflow`, `log_to_dvc`) |

For drift, `method="auto"` uses KS on smaller numeric samples and PSI on larger numeric samples. Use `method="wasserstein"` when a normalized distribution-distance signal is more useful than a hypothesis test.

## Screenshots

<p align="center">
  <img src="https://raw.githubusercontent.com/neoline361-art/fitcheck/main/assets/screenshots/terminal-check.png" width="32%" alt="Terminal check output">
  <img src="https://raw.githubusercontent.com/neoline361-art/fitcheck/main/assets/screenshots/html-report.png" width="32%" alt="Interactive HTML report">
  <img src="https://raw.githubusercontent.com/neoline361-art/fitcheck/main/assets/screenshots/demo-output.png" width="32%" alt="One-command demo">
</p>

Terminal check output (left), the interactive Plotly model report (center), and the one-command demo (right).

## Feature comparison

| Capability | FitCheck | Evidently | Deepchecks | Pandera |
|---|---|---|---|---|
| Data quality checks (missing, duplicates, constants, outliers) | ✅ | ✅ | ✅ | ✅ |
| Model evaluation (classification/regression) | ✅ | ✅ | ✅ | — |
| Drift detection (KS/PSI, categorical) | ✅ | ✅ | ✅ | — |
| CI-native CLI with exit codes (`--json`, `--fail-on`) | ✅ | ✅ | ✅ | ⚠️ |
| Plugin / custom check registry | ✅ | ✅ | ✅ | ✅ |
| Jupyter integration | ✅ | ✅ | ✅ | ✅ |
| Interactive HTML reports | ✅ | ✅ | ✅ | ⚠️ |
| DataFrame schema validation | ⚠️ | ⚠️ | ⚠️ | ✅ |
| Local-first, no telemetry, no hosted service | ✅ | ⚠️ | ⚠️ | ✅ |
| One-command `full` workflow (data + model + drift) | ✅ | — | — | — |

✅ = supported; ⚠️ = partial or via add-on; — = not a core feature. FitCheck's rows reflect the verified contents of this repository. Competitor rows are based on public documentation and may change — verify before making procurement decisions.

## CLI commands

```bash
fitcheck check data.csv --target label
fitcheck check data1.csv data2.csv            # multi-file check
fitcheck check data.csv --missing-warning 0.10 --missing-critical 0.30
fitcheck check data.csv --time-column timestamp --plugins my_checks
fitcheck check big.parquet --backend polars        # optional fast loading backend
fitcheck report model.joblib X_test.npy y_test.npy --renderer plotly
fitcheck drift train.csv production.csv --method psi
fitcheck full data.csv --target label --model model.joblib --reference train.csv
fitcheck demo --no-browser --output-dir ./demo
```

## CI and exit codes

`fitcheck check` is CI-native: run it with `--json` for machine-readable output, `--quiet` to suppress everything except the exit code, and `--fail-on` to pick the severity that fails a pipeline.

```bash
fitcheck check data.csv --target label --json --quiet --fail-on critical
echo $?   # 0 pass, 1 warnings, 2 critical, 3 runtime error
```

`--json` emits a result dict for a single file and a list of dicts for multiple files. High-cardinality detection targets ID-like columns (object, category, and integer dtypes), so continuous numeric measurements are not flagged as cardinality issues.

| Exit code | Meaning |
|---|---|
| `0` | No issues (or only issues below `--fail-on`) |
| `1` | Warnings found |
| `2` | Critical issues found |
| `3` | Runtime error (missing file, invalid config) |

The repository ships a [pre-commit hook](.pre-commit-hooks.yaml) and a GitHub Action gate ([`.github/workflows/fitcheck-gate.yml`](.github/workflows/fitcheck-gate.yml)) so dataset health blocks merges the same way linting does.

## Plugins and Jupyter

Custom checks are plain functions that take a DataFrame and return issue dictionaries:

```python
from fitcheck.plugins import registry

def my_check(df):
    return [{"column": "x", "type": "custom", "severity": "warning",
             "message": "custom rule triggered", "suggestion": "fix it"}]

registry.register("my_check", my_check)
```

```bash
fitcheck check data.csv --plugins my_check
```

Inside a notebook, `%load_ext fitcheck` enables inline reports:

```python
%load_ext fitcheck
%fitcheck df --target label
```

## Reports and privacy

FitCheck does not upload input data. HTML reports embed generated plots as base64 data and include only the information derived from the supplied datasets. Model loading uses Python pickle for user-owned artifacts; never load a model file from an untrusted source.

## Development and verification

```bash
pip install -r requirements.lock   # reproducible dev environment
pip install -e . --no-deps
ruff check fitcheck tests
mypy fitcheck
bandit -r fitcheck/ -x tests
pytest --cov=fitcheck --cov-report=term-missing
```

The current repository suite contains **80+ passing tests** and reports approximately **92% total coverage** on the supported Python environment.

## Large CSVs and contact data

Phone numbers should be stored as strings, not numeric values, so leading zeros and country prefixes are preserved. Names are treated as text values. A quick local check is:

```bash
fitcheck check contacts.csv --output contacts_report.html
# Fast, explicit sample review for a very large CSV:
fitcheck check contacts.csv --sample-rows 100000 --output contacts_sample_report.html
```

FitCheck is designed for in-memory pandas workflows. A file with 1 million rows and a few narrow columns is a reasonable local smoke-test target, but a 10-million-row file may require several gigabytes of RAM depending on string length and pandas version. For a fast schema/sample review, use pandas to create a representative sample before calling FitCheck; do not claim a sample report is a full-dataset audit. FitCheck does not print or transmit raw phone numbers in the terminal, but generated reports can contain previews, so protect report files as sensitive data.

## Documentation

| Resource | Purpose |
|---|---|
| [API reference](docs/API.md) | Public functions, arguments, and return values |
| [Architecture](docs/ARCHITECTURE.md) | Module boundaries and design principles |
| [FAQ](docs/FAQ.md) | Common questions and limitations |
| [Examples](examples/basic_usage.py) | Runnable Python examples |
| [Changelog](CHANGELOG.md) | Release history |

## License

FitCheck is released under the Apache 2.0 License. See [LICENSE](LICENSE).
