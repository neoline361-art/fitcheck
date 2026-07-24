"""Distribution drift detection — compare reference vs production datasets."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from fitcheck.html import render_drift_html


def detect_drift(
    reference: str | pd.DataFrame,
    production: str | pd.DataFrame,
    output: str = "drift_report.html",
    threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Detect distribution drift between reference and production data.

    Args:
        reference: Path to reference dataset or DataFrame.
        production: Path to production dataset or DataFrame.
        output: Path for the generated HTML report.
        threshold: P-value threshold for detecting drift (default 0.05).

    Returns:
        List of drift results per feature.
    """
    ref_df = _load_data(reference)
    prod_df = _load_data(production)

    common_cols = [c for c in ref_df.columns if c in prod_df.columns]
    results: list[dict[str, Any]] = []

    for col in common_cols:
        ref_col = ref_df[col].dropna()
        prod_col = prod_df[col].dropna()

        if ref_col.dtype.kind in "iufc" and ref_col.nunique() > 10:
            result = _ks_test(ref_col, prod_col, threshold)
        else:
            result = _chi2_test(ref_col, prod_col, threshold)

        result["feature"] = col
        results.append(result)

    render_drift_html(results, ref_df, prod_df, output)
    return results


def _load_data(data: str | pd.DataFrame) -> pd.DataFrame:
    """Load data from path or return DataFrame as-is."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if not os.path.exists(data):
        raise FileNotFoundError(f"Data file not found: {data}")
    if data.endswith(".parquet"):
        return pd.read_parquet(data)
    return pd.read_csv(data)


def _ks_test(
    ref_col: pd.Series, prod_col: pd.Series, threshold: float
) -> dict[str, Any]:
    """Kolmogorov-Smirnov test for numeric columns."""
    ref_vals = pd.to_numeric(ref_col, errors="coerce").dropna()
    prod_vals = pd.to_numeric(prod_col, errors="coerce").dropna()

    if len(ref_vals) == 0 or len(prod_vals) == 0:
        return {
            "type": "numeric",
            "test": "KS",
            "statistic": 0.0,
            "p_value": 1.0,
            "drifted": False,
            "severity": "info",
            "message": "Insufficient numeric data for KS test",
        }

    stat, p = stats.ks_2samp(ref_vals, prod_vals)
    drifted = p < threshold
    severity = "critical" if drifted else "info"
    return {
        "type": "numeric",
        "test": "KS",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drifted": drifted,
        "severity": severity,
        "message": (
            f"KS stat={stat:.4f}, p={p:.4f} — "
            f"{'drift detected' if drifted else 'no drift'}"
        ),
    }


def _chi2_test(
    ref_col: pd.Series, prod_col: pd.Series, threshold: float
) -> dict[str, Any]:
    """Chi-squared test for categorical columns."""
    ref_counts = ref_col.value_counts()
    prod_counts = prod_col.value_counts()
    all_cats = ref_counts.index.union(prod_counts.index)

    if len(all_cats) == 0:
        return {
            "type": "categorical",
            "test": "Chi2",
            "statistic": 0.0,
            "p_value": 1.0,
            "drifted": False,
            "severity": "info",
            "message": "No categories found for Chi2 test",
        }

    ref_vec = np.array([ref_counts.get(c, 0) for c in all_cats])
    prod_vec = np.array([prod_counts.get(c, 0) for c in all_cats])

    # Add pseudocount to avoid zero expected frequencies
    ref_vec = ref_vec + 0.5
    prod_vec = prod_vec + 0.5

    try:
        stat, p, _, _ = stats.chi2_contingency([ref_vec, prod_vec])
    except ValueError:
        return {
            "type": "categorical",
            "test": "Chi2",
            "statistic": 0.0,
            "p_value": 1.0,
            "drifted": False,
            "severity": "info",
            "message": "Chi2 test could not be computed",
        }

    drifted = p < threshold
    severity = "critical" if drifted else "info"
    return {
        "type": "categorical",
        "test": "Chi2",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drifted": drifted,
        "severity": severity,
        "message": (
            f"Chi2 stat={stat:.4f}, p={p:.4f} — "
            f"{'drift detected' if drifted else 'no drift'}"
        ),
    }
