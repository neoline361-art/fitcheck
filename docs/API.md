# API Reference

## `fitcheck.check(data, target=None, output="fitcheck_report.html", return_format="list", auto_fix=False)`

Validate a dataset's health.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `data` | `str` or `pd.DataFrame` | required | CSV/Parquet path or DataFrame |
| `target` | `str` or `None` | `None` | Target column for ML-specific checks |
| `output` | `str` | `"fitcheck_report.html"` | HTML report output path |
| `return_format` | `str` | `"list"` | `"list"`, `"dict"`, or `"json"` |
| `auto_fix` | `bool` | `False` | Generate fix script if issues found |

**Returns:** `list[dict]`, `dict`, or `str` — depending on `return_format`

**Raises:** `FileNotFoundError`, `ValueError`

**Checks performed:**
- Missing values (>5% warning, >20% critical)
- Duplicate rows
- Constant columns (zero variance)
- Class imbalance (majority >80%)
- Outliers (IQR method, >1%)

**Example:**
```python
issues = fitcheck.check("data.csv", target="label", auto_fix=True)
```

---

## `fitcheck.report(model, X_test, y_test, output="model_report.html")`

Evaluate a trained model.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | Any | required | Trained model with `.predict()` |
| `X_test` | `pd.DataFrame` or `np.ndarray` | required | Test features |
| `y_test` | `pd.Series` or `np.ndarray` | required | Test targets |
| `output` | `str` | `"model_report.html"` | HTML report path |

**Returns:** `dict` — computed metrics

**Auto-detects:** classification vs regression

**Classification metrics:** accuracy, precision, recall, F1, ROC-AUC (binary), confusion matrix

**Regression metrics:** MSE, RMSE, MAE, R2, residuals plot

**Example:**
```python
metrics = fitcheck.report(model, X_test, y_test)
print(metrics["accuracy"])
```

---

## `fitcheck.detect_drift(reference, production, output="drift_report.html", threshold=0.05)`

Detect distribution drift between two datasets.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `reference` | `str` or `pd.DataFrame` | required | Reference/training dataset |
| `production` | `str` or `pd.DataFrame` | required | Production dataset |
| `output` | `str` | `"drift_report.html"` | HTML report path |
| `threshold` | `float` | `0.05` | P-value threshold |

**Returns:** `list[dict]` — per-feature results with `drifted` flag

**Tests:** KS test (numeric, >10 unique values), Chi-squared (categorical)

**Example:**
```python
results = fitcheck.detect_drift("train.csv", "prod.csv")
drifted = sum(r["drifted"] for r in results)
```
