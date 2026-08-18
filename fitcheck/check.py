"""Dataset health check engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from fitcheck.html import render_check_html


def check(
    data: str | pd.DataFrame,
    target: str | None = None,
    output: str = "fitcheck_report.html",
    return_format: str = "list",
    auto_fix: bool = False,
    config: dict[str, float] | None = None,
    plugins: list[Any] | None = None,
    time_column: str | None = None,
    sample_rows: int | None = None,
    backend: str = "pandas",
) -> list[dict[str, Any]] | dict[str, Any] | str:
    """Run a comprehensive health check on a dataset.

    Args:
        data: Path to CSV/Parquet file or a pandas DataFrame.
        target: Name of the target column for ML-specific checks.
        output: Path for the generated HTML report.
        return_format: One of "list", "dict", or "json".
        auto_fix: If True, generate a Python fix script alongside the report.
        config: Optional threshold overrides; omitted values keep safe defaults.
        plugins: Optional custom check functions returning issue dictionaries.
        time_column: Optional timestamp column for basic time-series validation.
        sample_rows: Optional number of CSV rows to inspect instead of loading the full file.
        backend: Data loading backend: "pandas" (default), "polars", or "duckdb" (both optional dependencies).

    Returns:
        Issues found in the dataset. Format depends on return_format.

    Raises:
        FileNotFoundError: If data is a string path that does not exist.
        ValueError: If return_format is not recognized.
    """
    if sample_rows is not None and sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer")
    df = _load_data(data, sample_rows=sample_rows, backend=backend)
    input_path = data if isinstance(data, str) else "dataframe_input"
    if target and target not in df.columns:
        raise ValueError(f'Target column "{target}" not found in columns: {list(df.columns)}')

    thresholds: dict[str, float] = {
        "missing_critical": 0.20,
        "missing_warning": 0.05,
        "duplicate_threshold": 0.05,
        "imbalance_threshold": 0.80,
        "outlier_threshold": 0.01,
        "high_cardinality_ratio": 0.95,
        "text_length_outlier_multiplier": 3.0,
    }
    if config:
        unknown = set(config) - set(thresholds)
        if unknown:
            raise ValueError(f"Unknown check threshold(s): {', '.join(sorted(unknown))}")
        thresholds.update(config)
    if thresholds["missing_critical"] < thresholds["missing_warning"]:
        raise ValueError("missing_critical must be greater than or equal to missing_warning")
    config = thresholds

    issues: list[dict[str, Any]] = []
    issues.extend(_detect_missing(df, config))
    issues.extend(_detect_duplicates(df, config))
    issues.extend(_detect_constants(df))
    issues.extend(_detect_high_cardinality(df, config))
    issues.extend(_detect_text_length(df, config))
    issues.extend(_detect_text_encoding(df))
    if target and target in df.columns:
        issues.extend(_detect_imbalance(df, target, config))
        issues.extend(_detect_outliers(df, config, exclude=target))
    else:
        issues.extend(_detect_outliers(df, config))

    if plugins:
        from fitcheck.extensions import run_plugins

        issues.extend(run_plugins(df, plugins))
    if time_column:
        from fitcheck.extensions import validate_timeseries

        issues.extend(validate_timeseries(df, time_column))

    result_dict = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "sampled": sample_rows is not None and isinstance(data, str),
        "sample_rows": sample_rows if isinstance(data, str) else None,
        "issues": issues,
        "passed": len(issues) == 0,
        "config": config,
        "summary": {
            "critical": sum(1 for i in issues if i.get("severity") == "critical"),
            "warning": sum(1 for i in issues if i.get("severity") == "warning"),
            "info": sum(1 for i in issues if i.get("severity") == "info"),
        },
    }

    render_check_html(issues, df, output)

    if auto_fix and issues:
        try:
            from fitcheck.fix import generate_fix_script

            script_path = str(Path(output).with_suffix("")) + "_fix_script.py"
            generate_fix_script(result_dict, input_path, script_path)
        except ImportError:
            pass

    if return_format == "dict":
        return result_dict
    elif return_format == "json":
        return json.dumps(result_dict, indent=2, default=str)
    else:
        return issues


def _load_data(
    data: str | pd.DataFrame,
    sample_rows: int | None = None,
    backend: str = "pandas",
) -> pd.DataFrame:
    """Load data from path or return DataFrame as-is."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if not os.path.exists(data):
        raise FileNotFoundError(f"Data file not found: {data}")
    if backend in ("polars", "duckdb"):
        from fitcheck.backends import get_backend

        backend_obj = get_backend(backend)
        frame = backend_obj.read(data)
        df = backend_obj.to_pandas(frame)
        return df.head(sample_rows) if sample_rows is not None else df
    if data.endswith(".parquet"):
        return pd.read_parquet(data)
    return pd.read_csv(data, nrows=sample_rows)


def _detect_missing(df: pd.DataFrame, config: dict[str, float]) -> list[dict[str, Any]]:
    """Detect columns with missing values above thresholds."""
    issues = []
    for col in df.columns:
        pct = df[col].isnull().mean()
        if pct >= config["missing_critical"]:
            issues.append(
                {
                    "column": col,
                    "type": "missing_values",
                    "severity": "critical",
                    "message": f"{col}: {pct:.1%} missing (critical threshold: {config['missing_critical']:.0%})",
                    "suggestion": f'Consider dropping column or advanced imputation for "{col}"',
                }
            )
        elif pct >= config["missing_warning"]:
            issues.append(
                {
                    "column": col,
                    "type": "missing_values",
                    "severity": "warning",
                    "message": f"{col}: {pct:.1%} missing (warning threshold: {config['missing_warning']:.0%})",
                    "suggestion": f'Apply median/mode imputation for "{col}"',
                }
            )
    return issues


def _detect_duplicates(df: pd.DataFrame, config: dict[str, float]) -> list[dict[str, Any]]:
    """Detect duplicate rows."""
    issues = []
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        pct = dup_count / len(df)
        severity = "warning" if pct >= config["duplicate_threshold"] else "info"
        issues.append(
            {
                "column": "all",
                "type": "duplicate_rows",
                "severity": severity,
                "message": f"{dup_count} duplicate rows ({pct:.1%})",
                "suggestion": "Consider removing duplicate rows with df.drop_duplicates()",
            }
        )
    return issues


def _detect_constants(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect columns with a single unique value."""
    issues = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue  # fully missing columns are covered by missing-values detection
        if series.nunique() == 1:
            issues.append(
                {
                    "column": col,
                    "type": "constant_column",
                    "severity": "warning",
                    "message": f'{col}: constant value "{series.iloc[0]}" (zero variance)',
                    "suggestion": f'Drop constant column "{col}" — provides no information',
                }
            )
    return issues


def _detect_imbalance(
    df: pd.DataFrame, target: str, config: dict[str, float]
) -> list[dict[str, Any]]:
    """Detect class imbalance in the target column."""
    issues = []
    vc = df[target].value_counts(normalize=True)
    if len(vc) > 1 and vc.iloc[0] >= config["imbalance_threshold"]:
        issues.append(
            {
                "column": target,
                "type": "class_imbalance",
                "severity": "warning",
                "message": (
                    f"{target}: class imbalance — majority class is "
                    f'"{vc.index[0]}" at {vc.iloc[0]:.1%}'
                ),
                "suggestion": "Consider SMOTE, class weights, or stratified sampling",
            }
        )
    return issues


def _detect_outliers(
    df: pd.DataFrame,
    config: dict[str, float],
    exclude: str | None = None,
) -> list[dict[str, Any]]:
    """Detect outliers in numeric columns using the IQR method."""
    issues = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if exclude and exclude in numeric_cols:
        numeric_cols.remove(exclude)
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        q1, q3 = col_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = ((col_data < lower) | (col_data > upper)).sum()
        pct = outlier_count / len(col_data)
        if pct >= config["outlier_threshold"]:
            issues.append(
                {
                    "column": col,
                    "type": "outliers",
                    "severity": "info",
                    "message": f"{col}: {outlier_count} outliers ({pct:.1%}) via IQR",
                    "suggestion": f'Review extreme values in "{col}" or apply capping',
                }
            )
    return issues


def _detect_high_cardinality(df: pd.DataFrame, config: dict[str, float]) -> list[dict[str, Any]]:
    """Flag ID-like columns whose uniqueness ratio exceeds the configured threshold."""
    issues = []
    threshold = config["high_cardinality_ratio"]
    for col in df.columns:
        dtype = df[col].dtype
        # Continuous floats are expected to be ~unique; only ID-like dtypes are meaningful.
        if not (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_integer_dtype(dtype)
            or (
                hasattr(pd, "StringDtype")
                and pd.api.types.is_string_dtype(dtype)
                and not pd.api.types.is_object_dtype(dtype)
            )
        ):
            continue
        non_null = df[col].dropna()
        if len(non_null) < 50:
            continue  # cardinality is only meaningful past 50 rows (upgrade path: make configurable)
        ratio = non_null.nunique() / len(non_null)
        if ratio > threshold:
            issues.append(
                {
                    "column": col,
                    "type": "high_cardinality",
                    "severity": "info",
                    "message": f"{col}: {non_null.nunique()} unique values ({ratio:.1%} uniqueness)",
                    "suggestion": f'Verify "{col}" is not an ID column; group or hash high-cardinality values',
                }
            )
    return issues


def _detect_text_encoding(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag object columns with values that are not strictly UTF-8 encodable.

    Mixed encodings usually surface as lone surrogates after a lossy decode;
    a strict re-encode raises UnicodeEncodeError on those values.
    """
    issues = []
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(100)
        for value in sample:
            try:
                str(value).encode("utf-8", errors="strict")
            except UnicodeEncodeError:
                issues.append(
                    {
                        "column": col,
                        "type": "text_encoding",
                        "severity": "warning",
                        "message": f'{col}: contains non-UTF8 encodable characters (possible mixed encoding)',
                        "suggestion": f'Re-decode "{col}" with the correct source encoding before further analysis',
                    }
                )
                break
    return issues


def _safe_lengths(series: pd.Series) -> pd.Series:
    """Compute string lengths while normalizing lone surrogates safely.

    Uses a pure-Python scalar map so no value ever passes through pyarrow
    string conversion, which rejects lone surrogates at frame construction.
    """
    def _len(value: object) -> int:
        text = str(value).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        return len(text)

    return pd.Series([_len(v) for v in series], index=series.index, dtype="Int64")


def _detect_text_length(df: pd.DataFrame, config: dict[str, float]) -> list[dict[str, Any]]:
    """Flag object columns whose string lengths are strongly skewed (mean >> median)."""
    issues = []
    multiplier = config["text_length_outlier_multiplier"]
    for col in df.select_dtypes(include=["object"]).columns:
        lengths = _safe_lengths(df[col].dropna())
        if len(lengths) < 10:
            continue
        median = float(lengths.median())
        mean = float(lengths.mean())
        if mean > multiplier * max(median, 1.0):
            issues.append(
                {
                    "column": col,
                    "type": "text_length_outliers",
                    "severity": "info",
                    "message": f"{col}: mean length {mean:.1f} vs median {median:.1f} (possible outliers)",
                    "suggestion": f'Review unusually long strings in "{col}" or truncate/normalize',
                }
            )
    return issues
