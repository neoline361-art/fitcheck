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

    Returns:
        Issues found in the dataset. Format depends on return_format.

    Raises:
        FileNotFoundError: If data is a string path that does not exist.
        ValueError: If return_format is not recognized.
    """
    if sample_rows is not None and sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer")
    df = _load_data(data, sample_rows=sample_rows)
    input_path = data if isinstance(data, str) else "dataframe_input"

    thresholds: dict[str, float] = {
        "missing_critical": 0.20,
        "missing_warning": 0.05,
        "duplicate_threshold": 0.05,
        "imbalance_threshold": 0.80,
        "outlier_threshold": 0.01,
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


def _load_data(data: str | pd.DataFrame, sample_rows: int | None = None) -> pd.DataFrame:
    """Load data from path or return DataFrame as-is."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if not os.path.exists(data):
        raise FileNotFoundError(f"Data file not found: {data}")
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
        if df[col].nunique(dropna=False) == 1:
            issues.append(
                {
                    "column": col,
                    "type": "constant_column",
                    "severity": "warning",
                    "message": f'{col}: constant value "{df[col].iloc[0]}" (zero variance)',
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
