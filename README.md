<p align="center">
  <h1 align="center">FitCheck</h1>
  <p align="center"><em>Zero-boilerplate ML data validation and model evaluation.</em></p>
  <p align="center">
    <a href="https://github.com/neoline361-art/fitcheck/actions"><img src="https://img.shields.io/github/actions/workflow/status/neoline361-art/fitcheck/ci.yml?branch=main&logo=github&label=CI" alt="CI"></a>
    <a href="https://github.com/neoline361-art/fitcheck/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache 2.0"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+"></a>
    <a href="https://github.com/neoline361-art/fitcheck/actions"><img src="https://img.shields.io/badge/Tests-28%20passing-brightgreen" alt="Tests"></a>
    <a href="https://github.com/neoline361-art/fitcheck/actions"><img src="https://img.shields.io/badge/Coverage-82%25-brightgreen" alt="Coverage"></a>
    <a href="https://github.com/neoline361-art/fitcheck/blob/main/SECURITY.md"><img src="https://img.shields.io/badge/Security-Policy-blue" alt="Security"></a>
    <a href="https://github.com/neoline361-art/fitcheck/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/Changelog-v2.0.0-blue" alt="Changelog"></a>
  </p>
</p>

---

**FitCheck** validates datasets, evaluates ML models, and detects distribution drift — all with one-line commands that generate shareable HTML reports.

| Package | Release | Stats |
|---------|---------|-------|
| fitcheck | `pip install fitcheck` | ![Platform: Linux, macOS, Windows](https://img.shields.io/badge/platform-linux--macos--windows-lightgrey) |

## ✨ Features

### 🔍 Dataset Validation
- **Missing Value Detection** — Identifies columns with >5% (warning) or >20% (critical) null values
- **Duplicate Row Detection** — Finds exact duplicate rows in your dataset
- **Constant Column Detection** — Flags columns with only a single unique value
- **Class Imbalance Detection** — Warns when majority class exceeds 80%
- **Outlier Detection** — Uses IQR method (1.5×) to identify statistical outliers
- **Auto-Fix Scripts** — Generates executable Python scripts to fix detected issues

### 📊 Model Evaluation
- **Auto-Detection** — Automatically detects classification vs regression tasks
- **Classification Metrics** — Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression Metrics** — MSE, RMSE, MAE, R²
- **Visualizations** — Confusion matrix, ROC curve, residual plots
- **Feature Importance** — Extracts and displays feature importances for tree-based models

### 🔄 Drift Detection
- **Numeric Drift** — Kolmogorov-Smirnov test for continuous features
- **Categorical Drift** — Chi-squared test for categorical features
- **Configurable Thresholds** — Set custom p-value thresholds for drift detection

### 📱 Shareable Reports
- **Self-Contained HTML** — All reports are standalone HTML files ready for sharing
- **Interactive Visualizations** — Embedded charts and graphs
- **Slack/Email Ready** — Perfect for team communication and stakeholder updates

## 🎯 Philosophy

- **Zero Config** — Pass a file path. Get answers. No YAML, no setup.
- **Immutability** — FitCheck diagnoses, never silently modifies. Fix scripts are transparent and inspectable.
- **Shareability** — Every check generates a self-contained HTML report ready for Slack, email, or GitHub.
- **No Telemetry** — Zero outbound network calls. All computation is local.

## 📦 Installation

```bash
pip install fitcheck
```

From source:

```bash
git clone https://github.com/neoline361-art/fitcheck.git
cd fitcheck
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Python API

```python
import fitcheck

# 1. Validate a dataset
issues = fitcheck.check("data.csv", target="label", auto_fix=True)

# 2. Evaluate a model
metrics = fitcheck.report(model, X_test, y_test)

# 3. Detect drift
results = fitcheck.detect_drift("train.csv", "production.csv")

# 4. Generate fix script (Pro feature)
from fitcheck.pro import generate_fix_script
script = generate_fix_script(diagnostics, "data.csv", "fix_data.py")
```

### Command-Line Interface

```bash
# Validate dataset
fitcheck check data.csv --target label --auto-fix

# Evaluate model
fitcheck report model.pkl X_test.npy y_test.npy

# Detect drift
fitcheck drift train.csv production.csv --threshold 0.05

# Run demo
fitcheck demo
```

## 📋 What FitCheck Checks

| Check | Method | Severity |
|-------|--------|----------|
| Missing values | Null ratio >5% / >20% | Warning / Critical |
| Duplicate rows | `df.duplicated().sum()` | Warning |
| Constant columns | Single unique value | Warning |
| Class imbalance | Majority class >80% | Warning |
| Outliers | IQR method (1.5×) | Info |
| Numeric drift | KS test (p<0.05) | Critical |
| Categorical drift | Chi-squared (p<0.05) | Critical |

## 📖 Documentation

| Resource | Description |
|----------|-------------|
| [API Reference](docs/API.md) | Complete API documentation with parameters, returns, and examples |
| [Architecture](docs/ARCHITECTURE.md) | Module design and design decisions |
| [Design Decisions](docs/DECISIONS.md) | Why certain technical choices were made |
| [FAQ](docs/FAQ.md) | Frequently asked questions |
| [Examples](examples/basic_usage.py) | Runnable usage examples |
| [Benchmarks](benchmarks/results.md) | Performance benchmarks on standard hardware |

## 🧪 End-to-End Examples

### Example 1: Complete Data Pipeline

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import fitcheck

# Load and validate data
df = pd.read_csv("your_data.csv")
issues = fitcheck.check(df, target="churn", output="data_quality.html")

# If issues found, generate fix script
if issues:
    from fitcheck.pro import generate_fix_script
    generate_fix_script(issues, "your_data.csv", "fix_pipeline.py")
    # Run: python fix_pipeline.py

# Train model
X = df.drop(columns=["churn"])
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model
metrics = fitcheck.report(model, X_test, y_test, output="model_eval.html")
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Later: Monitor for drift
production_data = pd.read_csv("new_data.csv")
drift_results = fitcheck.detect_drift(X_train, production_data, output="drift_report.html")
drifted_features = sum(1 for r in drift_results if r["drifted"])
print(f"Drifted features: {drifted_features}")
```

### Example 2: CLI Workflow

```bash
# Step 1: Validate your dataset
fitcheck check customers.csv --target churn --auto-fix

# Step 2: Review the generated fix script
cat fitcheck_report_fix_script.py

# Step 3: Apply fixes
python fitcheck_report_fix_script.py

# Step 4: Train your model (your training script)
python train_model.py

# Step 5: Evaluate the trained model
fitcheck report model.pkl X_test.csv y_test.csv --output eval_report.html

# Step 6: Monitor production data for drift
fitcheck drift train_baseline.csv production_current.csv --threshold 0.01
```

### Example 3: Programmatic Usage with Custom Logic

```python
import fitcheck
from fitcheck.pro import FixScriptGenerator

# Custom validation workflow
diagnostics = fitcheck.check("data.parquet", target="label", return_format="dict")

# Access detailed diagnostics
for issue in diagnostics.get("issues", []):
    print(f"{issue['type']}: {issue['description']}")

# Generate custom fix script
generator = FixScriptGenerator(engine="pandas")
for issue in diagnostics.get("issues", []):
    action = generator._to_action(issue)
    if action:
        generator.add(action)

script = generator.generate("data.parquet", "cleaned_data.csv")
with open("custom_fix.py", "w") as f:
    f.write(script)
```

## 🔧 Development

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
python demo.py

# Run examples
python examples/basic_usage.py
```

### Quality Status

| Tool | Status | Notes |
|------|--------|-------|
| Tests | ✅ 28 passing | Full coverage of core functionality |
| Coverage | ✅ 82% | Core modules well-covered |
| Ruff | ✅ Clean | All linting issues resolved |
| mypy | ⚠️ 26 errors | Type checking needs improvement (pandas stubs) |
| bandit | ✅ Clean | No security issues |
| pip-audit | ✅ Clean | No known vulnerabilities |

## 🏗 Project Structure

```
fitcheck/
├── __init__.py      # Package exports (check, report, detect_drift)
├── __main__.py      # CLI entry point
├── check.py         # Dataset validation engine
├── cli.py           # Command-line interface
├── drift.py         # Distribution drift detection
├── fix.py           # Auto-fix script generation (Pro)
├── html.py          # HTML report rendering
├── pro/             # Pro features namespace
│   └── __init__.py  # Exports FixScriptGenerator
└── report.py        # Model evaluation engine
```

## 📊 Project Maturity

| Aspect | Status |
|--------|--------|
| Version | v2.0.0 — Semantic Versioning |
| Tests | 28 tests, 82% coverage |
| Type Safety | mypy strict mode (improving) |
| Linting | ruff clean |
| Security | bandit + pip-audit in CI |
| License | Apache 2.0 |
| Platforms | Linux, macOS, Windows (Python 3.9–3.13) |
| PyPI | Published |

## ⚠️ Limitations

- Supports only pandas DataFrames (CSV, Parquet)
- Drift: KS test for numeric, Chi-squared for categorical
- Deep learning model evaluation is not implemented
- Datasets must fit in memory
- No streaming or distributed processing
- Feature importance extraction limited to tree-based models

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). TL;DR:
1. Tests must pass
2. Ruff + mypy must be clean
3. Add CHANGELOG entry

### Areas for Contribution

- [ ] Improve mypy type annotations
- [ ] Add support for more file formats (Excel, JSON)
- [ ] Implement deep learning model evaluation
- [ ] Add streaming/distributed processing support
- [ ] Expand drift detection methods (PSI, Wasserstein distance)
- [ ] Add more visualization options
- [ ] Create Jupyter notebook widgets

## 🔒 Security

See [SECURITY.md](SECURITY.md). Report vulnerabilities to neoline361@gmail.com.

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with:
- [pandas](https://pandas.pydata.org/) — Data manipulation
- [scikit-learn](https://scikit-learn.org/) — ML metrics and utilities
- [matplotlib](https://matplotlib.org/) — Visualization
- [scipy](https://scipy.org/) — Statistical tests
- [Jinja2](https://jinja.palletsprojects.com/) — HTML templating
