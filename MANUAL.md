# FitCheck v2.0 — Complete Manual

## 1. Project Overview

| Field | Value |
|-------|-------|
| Name | FitCheck |
| Version | 2.0.0 |
| License | Apache 2.0 |
| Python | >= 3.9 |
| Tests | 28/28 passing, 82% coverage |
| Philosophy | Diagnose, don't operate. |

**What it does:** FitCheck validates ML datasets, evaluates models, and detects distribution drift — all with one-line commands that generate shareable HTML reports.

---

## 2. Installation & Setup

### 2.1 Fresh Clone & Install

```bash
# Clone your repo
git clone https://github.com/neoline361-art/fitcheck.git
cd fitcheck

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

### 2.2 Verify Installation

```bash
# Run all tests
pytest --cov=fitcheck --cov-report=term-missing

# Expected output: 28 passed, 82% coverage

# Run demo
python demo.py

# Expected: 3 HTML reports generated
```

---

## 3. Usage Guide

### 3.1 Python API (Recommended)

```python
import fitcheck
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- 1. Validate a dataset ---
df = pd.read_csv("your_data.csv")
issues = fitcheck.check(df, target="label", auto_fix=True)
# Generates: fitcheck_report.html + fitcheck_fix_script.py

# --- 2. Evaluate a model ---
model = RandomForestClassifier()
model.fit(X_train, y_train)
metrics = fitcheck.report(model, X_test, y_test)
# Generates: model_report.html with metrics + plots
# Auto-detects classification vs regression

# --- 3. Detect drift ---
results = fitcheck.detect_drift("train.csv", "production.csv")
# Generates: drift_report.html
# KS test for numeric, Chi-squared for categorical
```

### 3.2 Command Line

```bash
# Check dataset quality
fitcheck check data.csv --target label --output report.html --auto-fix

# Evaluate model (pickled model + numpy arrays)
fitcheck report model.pkl X_test.npy y_test.npy --output model_report.html

# Detect drift
fitcheck drift train.csv production.csv --threshold 0.05

# Run demo
fitcheck demo
```

### 3.3 Module Entry

```bash
python -m fitcheck check data.csv --target label
```

---

## 4. Return Formats

```python
# List format (default, backward compatible)
issues = fitcheck.check("data.csv")
# Returns: [{"column": "age", "type": "missing_values", "severity": "warning", ...}, ...]

# Dict format (for CI/integration)
result = fitcheck.check("data.csv", return_format="dict")
# Returns: {"total_rows": 1000, "total_columns": 15, "issues": [...], "passed": False, "summary": {...}}

# JSON format
json_str = fitcheck.check("data.csv", return_format="json")
# Returns: JSON string
```

---

## 5. Understanding the Reports

### 5.1 Dataset Check Report (`fitcheck_report.html`)

**Dark-mode HTML** with:
- Summary cards: rows, columns, critical/warning/info counts
- Per-issue breakdown with severity badges
- Suggested fixes for each issue
- Data preview table (first 10 rows)

**Detected issues:**
| Issue Type | When It Fires | Severity |
|------------|---------------|----------|
| `missing_values` | >5% null (warning), >20% null (critical) | warning/critical |
| `duplicate_rows` | Any duplicate rows detected | warning/info |
| `constant_column` | Column with only 1 unique value | warning |
| `class_imbalance` | Target class >80% of data | warning |
| `outliers` | >1% values outside IQR bounds | info |

### 5.2 Model Report (`model_report.html`)

**Auto-detects task type** and shows:
- Classification: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve
- Regression: MSE, RMSE, MAE, R2, residuals plot, actual vs predicted
- Feature importance (tree models)

### 5.3 Drift Report (`drift_report.html`)

- Per-feature statistical test results
- KS test for numeric features
- Chi-squared test for categorical features
- Color-coded drift/no-drift indicators

### 5.4 Fix Script (`*_fix_script.py`)

**The killer feature:** When `auto_fix=True`, FitCheck generates a **transparent, inspectable Python script** that:
- Has a WARNING header requiring manual review
- Loads data from INPUT_PATH (never overwrites input)
- Each fix as a commented step with rationale
- Saves cleaned data to a NEW file
- Prints a summary of all changes

**Workflow:**
```bash
# 1. Run check with auto_fix
python -c "import fitcheck; fitcheck.check('data.csv', auto_fix=True)"

# 2. Review the generated script
cat fitcheck_fix_script.py

# 3. Edit if needed, then run
python fitcheck_fix_script.py

# 4. Verify output
ls cleaned_data.csv
```

---

## 6. Git Workflow

### 6.1 Initial Push

```bash
# 1. Go to your local repo
cd /path/to/fitcheck

# 2. Create a fresh branch
git checkout -b v2.0-apache

# 3. Remove old files (keep only assets/, openhuman/, ponytail/ if needed)
rm -f fitcheck/*.py tests/*.py .github/workflows/*.yml pyproject.toml README.md demo.py LICENSE .gitignore

# 4. Copy all new files from output directory
cp -r /mnt/agents/output/fitcheck-v2/* .

# 5. Clean up generated artifacts (don't commit these)
rm -f *.html *_fix_script.py demo_data.csv .coverage
rm -rf .pytest_cache/ htmlcov/ __pycache__/ fitcheck/__pycache__/ tests/__pycache__/

# 6. Stage everything
git add .

# 7. Commit
git commit -m "FitCheck v2.0 - Complete rewrite

- Apache 2.0 license
- Full type hints (mypy strict mode)
- 28 tests, 82% coverage (pytest + pytest-cov)
- Auto-fix script generation (transparent, never silent)
- GitHub Actions CI (ruff + mypy + pytest on 5 Python versions)
- PR bot for data file validation
- Professional README with badges
- Zero-boilerplate API: check(), report(), detect_drift()"

# 8. Push
git push origin v2.0-apache

# 9. Create PR on GitHub, merge to main, then tag release
git checkout main
git pull origin main
git tag v2.0.0
git push origin v2.0.0
```

### 6.2 Verify CI is Working

After pushing, go to **GitHub > Actions** and verify:
- [ ] CI workflow passes on Python 3.9, 3.10, 3.11, 3.12, 3.13
- [ ] ruff lint passes
- [ ] mypy type check passes
- [ ] pytest with coverage passes

---

## 7. Social Media Launch Content

### 7.1 Twitter/X Post

```
Built FitCheck v2.0 — a zero-boilerplate ML data validation library.

One line to validate your dataset:
>>> fitcheck.check("data.csv", target="label", auto_fix=True)

It generates:
- Dark-mode HTML reports
- Transparent Python fix scripts (review before running)
- Model eval (auto-detects classification vs regression)
- Drift detection (KS + Chi-squared)

28 tests. 82% coverage. Apache 2.0.

github.com/neoline361-art/fitcheck

#MachineLearning #Python #DataScience #MLOps
```

### 7.2 LinkedIn Post

```
Just shipped FitCheck v2.0 — a Python library that validates ML datasets and evaluates models with one line of code.

The philosophy is simple: "Diagnose, don't operate."

FitCheck inspects your data, finds quality issues (missing values, duplicates, class imbalance, outliers, constant columns), and generates transparent HTML reports + Python fix scripts that you review before running.

It also evaluates trained models (auto-detecting classification vs regression) and detects distribution drift between training and production data.

Key details:
- 28 tests, 82% coverage
- Type-safe (mypy strict mode)
- Apache 2.0 license
- CI/CD with GitHub Actions
- Pre-commit hook support

Built with: pandas, scikit-learn, scipy, matplotlib, jinja2

Check it out: github.com/neoline361-art/fitcheck

#machinelearning #python #datascience #mlops #opensource
```

### 7.3 Reddit (r/MachineLearning) Post

```
[Project] FitCheck v2.0 — Zero-boilerplate ML data validation

Hi everyone,

I built FitCheck, a Python library that does dataset health checks, model evaluation, and drift detection — all with one-line commands that generate dark-mode HTML reports.

The "killer feature" is auto_fix: instead of silently modifying your data, FitCheck generates a transparent Python script with every fix step commented. You review it, then run it.

**Quick example:**
```python
import fitcheck

# Validates dataset, finds issues, generates fix script
issues = fitcheck.check("data.csv", target="label", auto_fix=True)
```

**What it checks:**
- Missing values (configurable thresholds)
- Duplicate rows
- Constant columns (zero variance)
- Class imbalance
- Outliers (IQR method)

**Model evaluation:**
- Auto-detects classification vs regression
- Classification: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Regression: MSE, RMSE, MAE, R2, residuals plot

**Drift detection:**
- KS test for numeric features
- Chi-squared for categorical

**Quality:**
- 28 tests, 82% coverage
- Full type hints, mypy strict clean
- ruff lint clean
- Apache 2.0

GitHub: https://github.com/neoline361-art/fitcheck

Would love feedback!
```

### 7.4 Hacker News (Show HN) Post

```
Show HN: FitCheck – Zero-boilerplate ML data validation library

FitCheck validates datasets, evaluates models, and detects distribution drift — one-line commands that generate shareable HTML reports.

The thing I'm most proud of: auto_fix generates transparent Python fix scripts instead of silently mutating your data. Every fix step is commented with rationale.

Example:
    import fitcheck
    issues = fitcheck.check("data.csv", target="label", auto_fix=True)
    # → fitcheck_report.html + fitcheck_fix_script.py

Tech: pandas, scikit-learn, scipy, matplotlib, jinja2
Quality: 28 tests, 82% coverage, mypy strict, ruff
License: Apache 2.0

GitHub: https://github.com/neoline361-art/fitcheck
```

---

## 8. File Structure (Final)

```
fitcheck/
├── fitcheck/
│   ├── __init__.py              # Public API exports
│   ├── __main__.py              # python -m fitcheck
│   ├── check.py                 # Dataset health engine (197 lines)
│   ├── report.py                # Model evaluation engine (180 lines)
│   ├── drift.py                 # Drift detection engine (154 lines)
│   ├── fix.py                   # Auto-fix script generator (230 lines)
│   ├── html.py                  # Dark-mode HTML rendering (245 lines)
│   ├── cli.py                   # Terminal interface (113 lines)
│   ├── .pre-commit-hooks.yaml   # Git pre-commit hook
│   └── pro/
│       └── __init__.py          # Pro module exports
├── tests/
│   ├── __init__.py
│   └── test_all.py              # 28 comprehensive tests (313 lines)
├── .github/
│   └── workflows/
│       ├── ci.yml               # Quality gates (ruff + mypy + pytest)
│       └── fitcheck-gate.yml    # PR bot for data file validation
├── pyproject.toml               # Modern packaging (hatchling)
├── demo.py                      # Quick demo script (78 lines)
├── README.md                    # Professional landing page (99 lines)
├── LICENSE                      # Apache 2.0 full text (201 lines)
└── .gitignore                   # Python standard (48 lines)
```

**Total: 15 source files, ~1,850 lines of code + docs**

---

## 9. Architecture Decisions

| Decision | Why |
|----------|-----|
| Separate modules (check/report/drift) | Clean separation = readable in hiring context |
| `__main__.py` | Standard Python convention, shows attention to detail |
| `.pre-commit-hooks.yaml` | Shows understanding of developer workflows |
| `demo.py` | One-run proof of concept, generates all 3 reports |
| Apache 2.0 | Corporate-friendly, patent protection |
| Hatchling | Modern, fast build backend |
| Dark-mode HTML | Professional look, shareable |
| Fix scripts instead of auto-mutation | Core philosophy: "Diagnose, don't operate" |

---

## 10. Next Steps (Post-Launch)

1. **Week 1:** Share on social media (Twitter, LinkedIn, Reddit, HN)
2. **Week 2:** Collect feedback, respond to GitHub issues
3. **Week 3:** Add features based on community requests
4. **Month 2:** Publish to PyPI (`pip install fitcheck`)
5. **Month 3:** Add more drift tests (PSI, Wasserstein), time-series support

---

*Generated for FitCheck v2.0 — Apache 2.0 — neoline361-art*
