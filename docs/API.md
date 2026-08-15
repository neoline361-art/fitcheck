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
| `config` | `dict[str, float]` or `None` | `None` | Overrides for `missing_warning`, `missing_critical`, `duplicate_threshold`, `imbalance_threshold`, `outlier_threshold`, `high_cardinality_ratio`, and `text_length_outlier_multiplier` |
| `plugins` | `list[Callable]` or `None` | `None` | Optional custom checks that return issue dictionaries |
| `time_column` | `str` or `None` | `None` | Optional timestamp column for ordering, parsing, duplicate, and frequency-gap checks |
| `sample_rows` | `int` or `None` | `None` | For CSV input, inspect only the first N rows and mark the result as sampled |

The default checks cover missing values, duplicate rows, constant columns, class imbalance, numeric IQR outliers, high cardinality, and text-length skew. Optional plugins and time-series checks are explicit additions. `sample_rows` is a memory-aware review, not a full-dataset audit; the returned dictionary records that distinction.

```python
result = fitcheck.check(
    "data.csv",
    target="label",
    return_format="dict",
    config={"missing_warning": 0.10, "missing_critical": 0.30},
)
```

## `fitcheck.check(...)` — extra parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `str` | `pandas` | `pandas` (default) or `polars` (optional dependency; faster loading of large CSV/Parquet files). |

The polars backend accelerates the load step and converts frames to pandas for the check engine; polars-native checks are a future optimisation.

## `fitcheck.detect_seasonality(series, period=None)`

Return an info-level issue dict when a numeric series shows repeatable seasonal autocorrelation at a candidate lag (`7`, `12`, `24`, or `30`, or an explicit `period`). Returns `None` when the series is too short or no pattern is found. Autocorrelation is a lightweight stand-in for STL decomposition, which is the documented upgrade path.

```python
issue = fitcheck.detect_seasonality(df["sales"], period=7)
```

## `fitcheck.get_backend(name=None, df=None)`

Return `PandasBackend` or `PolarsBackend` (auto-selected when `df` is already a polars frame).

## `fitcheck.get_renderer(name="static")`

Return the static (matplotlib) or plotly renderer. The plotly renderer requires `pip install data-fitcheck[plotly]`.

## `fitcheck.log_to_mlflow(result, run_id=None)` / `fitcheck.log_to_dvc(result, stage="validate", path="fitcheck_metrics.yaml")`

Optional callbacks that log a check result dict to the active MLflow run or write DVC-compatible YAML metrics. Both no-op (returning `False`) when their optional dependency is missing.

## `fitcheck.report(model, X_test, y_test, output="model_report.html", renderer="static")`

Evaluate a trained model and write an HTML report. A pandas DataFrame is passed through to the model so feature-name-aware scikit-learn estimators do not emit avoidable warnings.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | Any | required | Trained object with `.predict()` |
| `X_test` | `pd.DataFrame` or `np.ndarray` | required | Test features |
| `y_test` | `pd.Series` or `np.ndarray` | required | Test targets |
| `output` | `str` | `model_report.html` | HTML report path |

Classification reports include accuracy, weighted precision/recall/F1, support, confusion matrix, and, for binary probabilistic models, ROC-AUC, average precision, Brier score, ROC, precision–recall and calibration curves, a recommended threshold, and per-class error rates. Regression reports include MSE, RMSE, MAE, R², adjusted R², explained variance, residuals, and actual-versus-predicted plots. Feature importance uses tree attributes when present and falls back to SHAP values when `pip install data-fitcheck[shap]` is installed. With `renderer="plotly"` (requires the optional plotly extra), confusion-matrix and ROC charts render as interactive charts instead of static images.

```python
metrics = fitcheck.report(model, X_test, y_test)
print(metrics["accuracy"])
```

## `fitcheck.detect_drift(reference, production, output="drift_report.html", threshold=0.05, method="auto", psi_threshold=0.20, wasserstein_threshold=0.10, js_threshold=0.10)`

Compare common features between a reference dataset and production data. Schema drift — columns missing on either side and dtype family changes — is always reported as critical.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reference` | `str` or `pd.DataFrame` | required | Training or reference dataset |
| `production` | `str` or `pd.DataFrame` | required | Production dataset |
| `output` | `str` | `drift_report.html` | HTML report path |
| `threshold` | `float` | `0.05` | P-value threshold for KS and Chi-squared |
| `method` | `str` | `auto` | `auto`, `ks`, `psi`, `wasserstein`, `chi2`, or `js` |
| `psi_threshold` | `float` | `0.20` | PSI drift threshold |
| `wasserstein_threshold` | `float` | `0.10` | Normalized Wasserstein threshold |
| `js_threshold` | `float` | `0.10` | Jensen–Shannon divergence threshold |

Automatic selection uses KS for smaller numeric samples, PSI for larger numeric samples, and Chi-squared for categorical features. The normalized Wasserstein statistic divides distance by the reference standard deviation, making it easier to compare across feature scales.

```python
results = fitcheck.detect_drift("train.csv", "prod.csv", method="auto")
drifted = sum(result["drifted"] for result in results)
```

## CLI examples

```bash
fitcheck check data.csv --target label
fitcheck check big.parquet --backend polars          # optional fast loading
fitcheck report model.joblib X.npy y.npy --renderer plotly
fitcheck demo --no-browser --output-dir ./demo
fitcheck full data.csv --target label --model model.joblib --reference train.csv --output-dir reports
```

The full command writes dataset and model reports and adds a drift report when `--reference` is provided; `--model` is optional. It always writes an executive `index.html` linking the three reports. It uses all columns except the target as model features.

## Plugins

```python
from fitcheck.plugins import registry, load_plugin

registry.register("domain", my_check)          # named registration
check("data.csv", plugins=[load_plugin("domain")])
# or resolve a dotted module path:
check("data.csv", plugins=[load_plugin("my_pkg.my_checks")])
```

## Exit codes

`fitcheck check` returns `0` (pass), `1` (warnings), `2` (critical), or `3` (runtime error). `--fail-on` sets the minimum severity that fails the run; `--json` emits results to stdout; `--quiet` suppresses everything except JSON and the exit code.

## Jupyter

```bash
pip install "data-fitcheck[jupyter]"
```

```python
%load_ext fitcheck
%fitcheck df --target label          # line magic
%%fitcheck --target label            # cell magic, uses the df variable
```
