<h1 align="center">FitCheck</h1>
<p align="center"><em>Zero-boilerplate ML data validation, model evaluation, and drift detection.</em></p>
<p align="center">
  <a href="https://github.com/neoline361-art/fitcheck/actions"><img src="https://img.shields.io/github/actions/workflow/status/neoline361-art/fitcheck/ci.yml?branch=main&logo=github&label=CI" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/neoline361-art/fitcheck/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache 2.0"></a>
  <a href="https://github.com/neoline361-art/fitcheck/actions"><img src="https://img.shields.io/badge/Tests-260%20passing-brightgreen" alt="Tests"></a>
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

FitCheck ships a reproducible benchmark runner. Every measurement is an actual locally measured result with three warm-up runs followed by ten timed repetitions (median, minimum, and maximum reported); competitor timings appear only when those frameworks are measured on the same hardware, dataset, and workload definition, never as estimates typed into this file.

```bash
python benchmarks/run.py          # FitCheck workloads
python benchmarks/run.py --all    # FitCheck + importable competitors
make benchmark                    # same, via Makefile
```

Results accumulate in [benchmarks/results.md](benchmarks/results.md) with the full environment recorded (OS, CPU, RAM, Python, and framework versions) so a clean checkout can reproduce every number. See the methodology note at the top of that file.

<p align="center">
  <img src="https://raw.githubusercontent.com/neoline361-art/fitcheck/main/assets/charts/fitcheck_benchmark_comparison.png" width="85%" alt="Benchmark comparison and feature scorecard">
</p>

*Evidently and Deepchecks timings in the chart are community-reported estimates, not measured here — only the FitCheck numbers are measured. The feature-scorecard values are subjective availability scores; those tools offer deeper production-monitoring features.*

FitCheck is intentionally lightweight: it targets the common daily checks (missing values, duplicates, drift, model health) in seconds, not full production monitoring. For comprehensive statistical suites and richer monitoring, [Evidently](https://github.com/evidentlyai/evidently) and [Deepchecks](https://github.com/deepchecks/deepchecks) are excellent tools — FitCheck is the fast pre-flight check you run before every training run.

All backends produce identical diagnostics. The optional polars (`--backend polars`) and duckdb (`--backend duckdb`) backends accelerate CSV/Parquet loading for large files — they load and convert to pandas for analysis, so the speedup grows with the cost of CSV/Parquet loading. duckdb's auto type-inference is stricter than pandas, so the default pandas backend remains the safe choice for messy CSVs.

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
| Decision engine | Issue clustering by impact area, severity-weighted scoring (1–10), PASS/WARN/BLOCK verdicts, root-cause recommendations, YAML policy overrides |
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
| Decision engine (PASS/WARN/BLOCK, clustering, root-cause) | ✅ | — | — | — |
| Model evaluation (classification/regression) | ✅ | ✅ | ✅ | — |
| Drift detection (KS/PSI, categorical) | ✅ | ✅ | ✅ | — |
| CI-native CLI with exit codes (`--json`, `--fail-on`) | ✅ | ✅ | ✅ | ⚠️ |
| Plugin / custom check registry | ✅ | ✅ | ✅ | ✅ |
| Jupyter integration | ✅ | ✅ | ✅ | ✅ |
| Interactive HTML reports | ✅ | ✅ | ✅ | ⚠️ |
| DataFrame schema validation | ⚠️ | ⚠️ | ⚠️ | ✅ |
| Local-first, no telemetry, no hosted service | ✅ | ⚠️ | ⚠️ | ✅ |
| One-command `full` workflow (data + model + drift) | ✅ | — | — | — |
| Tamper-evident reports (SHA-256 fingerprint + HMAC signing) | ✅ | — | — | — |
| CLI report verification (`fitcheck verify`) | ✅ | — | — | — |

✅ = supported; ⚠️ = partial or via add-on; — = not a core feature. FitCheck's rows reflect the verified contents of this repository. Competitor rows are based on public documentation and may change — verify before making procurement decisions.

## CLI commands

```bash
fitcheck check data.csv --target label
fitcheck check data.csv --mode decision            # PASS/WARN/BLOCK verdict
fitcheck check data.csv --mode decision --policy fitcheck.yaml  # custom policy
fitcheck check data1.csv data2.csv            # multi-file check
fitcheck check data.csv --missing-warning 0.10 --missing-critical 0.30
fitcheck check data.csv --time-column timestamp --plugins my_checks
fitcheck check big.parquet --backend polars        # optional fast loading backend
fitcheck check big.parquet --backend duckdb         # optional out-of-core loading backend
fitcheck check data.csv --sign-key $SECRET          # HMAC-signed report
fitcheck check data.csv --artifact report.fitcheck.zip  # bundle report + fingerprint + signature
fitcheck verify report.html --against data.csv      # verify report integrity
fitcheck verify report.fitcheck.zip --against data.csv  # verify artifact bundle
fitcheck report model.joblib X_test.npy y_test.npy --renderer plotly
fitcheck drift train.csv production.csv --method psi
fitcheck full data.csv --target label --model model.joblib --reference train.csv
fitcheck demo --no-browser --output-dir ./demo
fitcheck doctor                   # diagnose the environment
fitcheck doctor --json            # machine-readable diagnosis
```

## CI and exit codes

`fitcheck check` is CI-native: run it with `--json` for machine-readable output, `--quiet` to suppress everything except the exit code, and `--fail-on` to pick the severity that fails a pipeline. The exit-code contract is verified end to end against datasets at every severity level:

| Exit code | Meaning | Verified against |
|---|---|---|
| `0` | No issues (or only issues below `--fail-on`) | clean dataset |
| `1` | Warnings found | 8% missing values (above the 5% warning band) |
| `2` | Critical issues found | 25% missing values (above the 20% critical band) |
| `3` | Runtime error (missing file, invalid config) | nonexistent file |

```bash
fitcheck check data.csv --target label --json --quiet --fail-on critical
echo $?   # 0 pass, 1 warnings, 2 critical, 3 runtime error
```

`--json` emits a result dict for a single file and a list of dicts for multiple files. High-cardinality detection targets ID-like columns (object, category, and integer dtypes), so continuous numeric measurements are not flagged as cardinality issues.

The repository ships a [pre-commit hook](.pre-commit-hooks.yaml) and a GitHub Action ([`action.yml`](action.yml)) so dataset health blocks merges the same way linting does.

### GitHub Action

Add FitCheck as a CI gate in your workflow:

```yaml
- name: FitCheck data validation
  uses: neoline361-art/fitcheck@v4.0.0
  with:
    command: 'check data.csv --mode decision'
    fail-on: 'warning'
    policy: 'fitcheck.yaml'
    secret-key: ${{ secrets.FITCHHECK_SECRET_KEY }}
```

| Input | Description | Default |
|---|---|---|
| `command` | FitCheck CLI command | `check data.csv --mode decision` |
| `fail-on` | Minimum severity to fail | `warning` |
| `policy` | Path to fitcheck.yaml | (empty) |
| `secret-key` | HMAC signing key | (empty) |

**Outputs:** `verdict` (PASS/WARN/BLOCK) and `exit-code` (0–3).

## Plugins and Jupyter

Custom checks are plain functions that take a DataFrame and return issue dictionaries, or structured `BaseCheck` subclasses:

```python
from fitcheck.plugins import registry

def my_check(df):
    return [{"column": "x", "type": "custom", "severity": "warning",
             "message": "custom rule triggered", "suggestion": "fix it"}]

registry.register("my_check", my_check)
```

Or use the structured `BaseCheck` contract for versioned, self-describing plugins:

```python
from fitcheck.plugins import BaseCheck, registry

class RangeCheck(BaseCheck):
    @property
    def name(self) -> str:
        return "range_check"

    @property
    def version(self) -> str:
        return "1.0.0"

    def run(self, df, config):
        issues = []
        for col in df.select_dtypes(include="number").columns:
            if df[col].max() > 1e6:
                issues.append({"column": col, "type": "range", "severity": "warning",
                               "message": f"{col} has values > 1e6"})
        return issues

registry.register("range_check", RangeCheck)
```

```bash
fitcheck check data.csv --plugins my_check
fitcheck check data.csv --plugins my_check,range_check  # multiple plugins
```

Inside a notebook, `%load_ext fitcheck` enables inline reports:

```python
%load_ext fitcheck
%fitcheck df --target label
```

## Trust & verification

Every FitCheck HTML report embeds a **visible cryptographic fingerprint** in the footer — dataset SHA-256, config hash, version, and timestamp. When signing is enabled, an HMAC-SHA256 signature is included. This makes reports **tamper-evident evidence**, not decoration.

### Verify a report

```bash
fitcheck verify fitcheck_report.html --against data.csv
# ✅ VALID — Report matches current data
# or
# ❌ TAMPERED — MISMATCH — report may be tampered or data has changed
```

### Sign reports with HMAC-SHA256

```bash
# Via CLI flag
fitcheck check data.csv --sign-key $FITCHECK_SECRET

# Via environment variable
export FITCHECK_SECRET_KEY=my-secret
fitcheck check data.csv

# Verify signature
fitcheck verify report.html --against data.csv --secret-key $FITCHECK_SECRET
```

### Why open-source + verifiable > closed-source + unverifiable

FitCheck is Apache 2.0 licensed — the same license as TensorFlow, PyTorch, and Linux. Readable code is a feature, not a bug. The real protection against tampering is not hiding source code; it is making every report **cryptographically tied to the exact data, configuration, and version** that produced it. No other tool in this space offers verifiable reports.

| Threat | Protection |
|---|---|
| Report edited after generation | Dataset SHA-256 mismatch detected |
| Report generated from different data | File hash comparison via `--against` |
| HMAC signature forged | Timing-safe comparison via `hmac.compare_digest` |
| Signature verified with wrong key | Exit code 1 + explicit error message |

### Verify an artifact bundle

```bash
# Create a bundle
fitcheck check data.csv --artifact report.fitcheck.zip --sign-key $SECRET

# Verify the bundle (data hash + HMAC signature)
fitcheck verify report.fitcheck.zip --against data.csv --secret-key $SECRET
# ✅ VALID — Bundle matches source data

# Or via Python API
from fitcheck.fingerprint import verify_report
result = verify_report("report.fitcheck.zip", "data.csv", secret_key="my-key")
print(result["match"])           # True or False
print(result["signature_valid"])  # True or False
```

### Python API

```python
from fitcheck.fingerprint import verify_report, hash_file

# Verify a report matches its source CSV
result = verify_report("report.html", "data.csv")
print(result["match"])       # True or False
print(result["message"])     # Human-readable result

# Verify with HMAC signature
result = verify_report("report.html", "data.csv", secret_key="my-key")
print(result["signature_valid"])  # True or False

# Hash a file directly
print(hash_file("data.csv"))  # SHA-256 hex digest
```

## Reports and privacy

FitCheck does not upload input data. HTML reports embed generated plots as base64 data and include only the information derived from the supplied datasets. Model loading uses Python pickle for user-owned artifacts; never load a model file from an untrusted source.

## Development and verification

The developer workflow is Makefile-driven; `make help` lists every target.

```bash
make install       # pip install -e .
make test          # full suite with coverage
make lint          # ruff check
make typecheck     # strict mypy
make security      # bandit
make audit         # pip-audit on runtime dependencies only
make doctor        # fitcheck doctor
make benchmark     # reproducible benchmark suite
make clean         # remove build and cache artifacts
```

The suite contains **260 passing tests** at approximately **95% total coverage**, with core mutation testing performed on the check engine (`fitcheck/check.py`) to drive test quality beyond line coverage. Reports are fully self-contained (no external CDN), responsive on narrow viewports, and use collapsible sections for large datasets.

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
