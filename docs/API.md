# FitCheck API Reference

## `fitcheck.check(data, target=None, output="fitcheck_report.html", return_format="list", auto_fix=False, config=None, plugins=None, time_column=None, sample_rows=None)`

Validate dataset health without mutating the input.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `str` or `pd.DataFrame` | required | CSV/Parquet path or DataFrame |
| `target` | `str` or `None` | `None` | Target column for imbalance and target-aware outlier checks |
| `output` | `str` | `fitcheck_report.html` | Self-contained HTML report path |
| `return_format` | `str` | `list` | `list`, `dict`, or `json` |
| `auto_fix` | `bool` | `False` | Generate a transparent Python fix script when issues exist |
| `config` | `dict[str, float]` or `None` | `None` | Overrides for `missing_warning`, `missing_critical`, `duplicate_threshold`, `imbalance_threshold`, and `outlier_threshold` |
| `plugins` | `list[Callable]` or `None` | `None` | Optional custom checks that return issue dictionaries |
| `time_column` | `str` or `None` | `None` | Optional timestamp column for ordering, parsing, and duplicate checks |
| `sample_rows` | `int` or `None` | `None` | For CSV input, inspect only the first N rows and mark the result as sampled |

The default checks cover missing values, duplicate rows, constant columns, class imbalance, and numeric IQR outliers. Optional plugins and time-series checks are explicit additions. `sample_rows` is a memory-aware review, not a full-dataset audit; the returned dictionary records that distinction.

```python
result = fitcheck.check(
    "data.csv",
    target="label",
    return_format="dict",
    config={"missing_warning": 0.10, "missing_critical": 0.30},
)
```

## `fitcheck.report(model, X_test, y_test, output="model_report.html")`

Evaluate a trained model and write an HTML report. A pandas DataFrame is passed through to the model so feature-name-aware scikit-learn estimators do not emit avoidable warnings.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | Any | required | Trained object with `.predict()` |
| `X_test` | `pd.DataFrame` or `np.ndarray` | required | Test features |
| `y_test` | `pd.Series` or `np.ndarray` | required | Test targets |
| `output` | `str` | `model_report.html` | HTML report path |

Classification reports include accuracy, weighted precision/recall/F1, support, confusion matrix, and, for binary probabilistic models, ROC-AUC, average precision, ROC and precision–recall curves, and a recommended threshold. Regression reports include MSE, RMSE, MAE, R², residuals, and actual-versus-predicted plots. Tree feature importance is included when exposed by the model.

```python
metrics = fitcheck.report(model, X_test, y_test)
print(metrics["accuracy"])
```

## `fitcheck.detect_drift(reference, production, output="drift_report.html", threshold=0.05, method="auto", psi_threshold=0.20, wasserstein_threshold=0.10)`

Compare common features between a reference dataset and production data.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reference` | `str` or `pd.DataFrame` | required | Training or reference dataset |
| `production` | `str` or `pd.DataFrame` | required | Production dataset |
| `output` | `str` | `drift_report.html` | HTML report path |
| `threshold` | `float` | `0.05` | P-value threshold for KS and Chi-squared |
| `method` | `str` | `auto` | `auto`, `ks`, `psi`, `wasserstein`, or `chi2` |
| `psi_threshold` | `float` | `0.20` | PSI drift threshold |
| `wasserstein_threshold` | `float` | `0.10` | Normalized Wasserstein threshold |

Automatic selection uses KS for smaller numeric samples, PSI for larger numeric samples, and Chi-squared for categorical features. The normalized Wasserstein statistic divides distance by the reference standard deviation, making it easier to compare across feature scales.

```python
results = fitcheck.detect_drift("train.csv", "prod.csv", method="auto")
drifted = sum(result["drifted"] for result in results)
```

## CLI full workflow

```bash
fitcheck full data.csv --target label --model model.joblib --reference train.csv --output-dir reports
```

The full command writes dataset and model reports and adds a drift report when `--reference` is provided. It uses all columns except the target as model features.
