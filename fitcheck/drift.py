from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from fitcheck.html import render_drift_html

DriftMethod = Literal["auto", "ks", "psi", "wasserstein", "chi2", "js"]


def detect_drift(
    reference: str | pd.DataFrame,
    production: str | pd.DataFrame,
    output: str = "drift_report.html",
    threshold: float = 0.05,
    method: DriftMethod = "auto",
    psi_threshold: float = 0.20,
    wasserstein_threshold: float = 0.10,
    js_threshold: float = 0.10,
) -> list[dict[str, Any]]:
    """Detect distribution drift between reference and production data.

    ``auto`` uses KS for small numeric samples, PSI for larger numeric samples,
    and Chi-squared for categorical features. Wasserstein and Jensen–Shannon can
    be selected explicitly. Schema drift (missing columns and dtype changes) is
    always reported as critical.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if method not in {"auto", "ks", "psi", "wasserstein", "chi2", "js"}:
        raise ValueError(f"Unsupported drift method: {method}")

    ref_df = _load_data(reference)
    prod_df = _load_data(production)
    results: list[dict[str, Any]] = _schema_drift(ref_df, prod_df)
    common_cols = [c for c in ref_df.columns if c in prod_df.columns]

    for col in common_cols:
        ref_col = ref_df[col].dropna()
        prod_col = prod_df[col].dropna()
        numeric = pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(prod_col)
        selected = _select_method(method, numeric, len(ref_col), len(prod_col))
        if selected == "ks":
            result = _ks_test(ref_col, prod_col, threshold)
        elif selected == "psi":
            result = _psi_test(ref_col, prod_col, psi_threshold)
        elif selected == "wasserstein":
            result = _wasserstein_test(ref_col, prod_col, wasserstein_threshold)
        elif selected == "js":
            result = _js_test(ref_col, prod_col, js_threshold)
        else:
            result = _chi2_test(ref_col, prod_col, threshold)
        result["feature"] = col
        results.append(result)

    render_drift_html(results, ref_df, prod_df, output)
    return results


def _select_method(method: DriftMethod, numeric: bool, ref_size: int, prod_size: int) -> str:
    if method != "auto":
        return "chi2" if method == "chi2" else method
    if not numeric:
        return "chi2"
    return "ks" if min(ref_size, prod_size) < 1_000 else "psi"


def _schema_drift(ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Report missing columns and dtype changes as critical schema drift."""
    results: list[dict[str, Any]] = []
    for col in ref_df.columns:
        if col not in prod_df.columns:
            results.append(_schema_result(col, f"Column {col!r} present in reference but missing in production"))
            continue
        ref_dtype, prod_dtype = ref_df[col].dtype, prod_df[col].dtype
        if _dtype_kind(ref_dtype) != _dtype_kind(prod_dtype):
            results.append(_schema_result(col, f"Column {col!r} dtype changed from {ref_dtype} to {prod_dtype}"))
    for col in prod_df.columns:
        if col not in ref_df.columns:
            results.append(_schema_result(col, f"Column {col!r} present in production but missing in reference"))
    return results


def _schema_result(feature: str, message: str) -> dict[str, Any]:
    return {
        "feature": feature,
        "type": "schema",
        "test": "Schema",
        "statistic": 0.0,
        "p_value": None,
        "drifted": True,
        "severity": "critical",
        "message": message,
        "suggestion": "Align the production schema with the reference schema before comparing distributions",
    }


def _dtype_kind(dtype: Any) -> str:
    """Broad dtype family so float32/float64 and int8/int64 do not false-positive."""
    kind = getattr(dtype, "kind", "O")
    if kind in "iuf":
        return "numeric"
    if kind == "b":
        return "bool"
    if kind in "M":
        return "datetime"
    if kind == "O":
        return "object"
    return str(kind)


def _load_data(data: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if not os.path.exists(data):
        raise FileNotFoundError(f"Data file not found: {data}")
    if data.endswith(".parquet"):
        return pd.read_parquet(data)
    return pd.read_csv(data)


def _empty_result(kind: str, test: str, message: str) -> dict[str, Any]:
    return {
        "type": kind,
        "test": test,
        "statistic": 0.0,
        "p_value": 1.0,
        "drifted": False,
        "severity": "info",
        "message": message,
    }


def _ks_test(ref_col: pd.Series, prod_col: pd.Series, threshold: float) -> dict[str, Any]:
    ref_vals = pd.to_numeric(ref_col, errors="coerce").dropna()
    prod_vals = pd.to_numeric(prod_col, errors="coerce").dropna()
    if len(ref_vals) == 0 or len(prod_vals) == 0:
        return _empty_result("numeric", "KS", "Insufficient numeric data for KS test")
    stat, p = stats.ks_2samp(ref_vals, prod_vals)
    drifted = bool(p < threshold)
    return {
        "type": "numeric", "test": "KS", "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4), "drifted": drifted,
        "severity": "critical" if drifted else "info",
        "message": f"KS stat={stat:.4f}, p={p:.4f} — {'drift detected' if drifted else 'no drift'}",
    }


def _psi_test(ref_col: pd.Series, prod_col: pd.Series, threshold: float) -> dict[str, Any]:
    ref = pd.to_numeric(ref_col, errors="coerce").dropna().to_numpy(dtype=float)
    prod = pd.to_numeric(prod_col, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(prod) == 0 or np.ptp(ref) == 0:
        return _empty_result("numeric", "PSI", "Insufficient variation for PSI")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return _empty_result("numeric", "PSI", "Insufficient variation for PSI")
    ref_bins = np.histogram(ref, bins=edges)[0].astype(float)
    prod_bins = np.histogram(np.clip(prod, edges[0], edges[-1]), bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_bins / len(ref), 1e-6, None)
    prod_pct = np.clip(prod_bins / len(prod), 1e-6, None)
    psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
    drifted = psi >= threshold
    return {
        "type": "numeric", "test": "PSI", "statistic": round(psi, 4),
        "p_value": None, "drifted": drifted,
        "severity": "critical" if drifted else "info",
        "message": f"PSI={psi:.4f} — {'drift detected' if drifted else 'no drift'}",
    }


def _wasserstein_test(ref_col: pd.Series, prod_col: pd.Series, threshold: float) -> dict[str, Any]:
    ref = pd.to_numeric(ref_col, errors="coerce").dropna().to_numpy(dtype=float)
    prod = pd.to_numeric(prod_col, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(prod) == 0:
        return _empty_result("numeric", "Wasserstein", "Insufficient numeric data")
    scale = float(np.std(ref)) or 1.0
    distance = float(stats.wasserstein_distance(ref, prod) / scale)
    drifted = distance >= threshold
    return {
        "type": "numeric", "test": "Wasserstein", "statistic": round(distance, 4),
        "p_value": None, "drifted": drifted,
        "severity": "critical" if drifted else "info",
        "message": f"Normalized distance={distance:.4f} — {'drift detected' if drifted else 'no drift'}",
    }


def _js_test(ref_col: pd.Series, prod_col: pd.Series, threshold: float) -> dict[str, Any]:
    """Jensen–Shannon divergence between two numeric distributions (base 2)."""
    ref = pd.to_numeric(ref_col, errors="coerce").dropna().to_numpy(dtype=float)
    prod = pd.to_numeric(prod_col, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(prod) == 0 or np.ptp(ref) == 0:
        return _empty_result("numeric", "JS", "Insufficient variation for Jensen–Shannon")
    bins = np.unique(np.quantile(np.concatenate([ref, prod]), np.linspace(0, 1, 11)))
    if len(bins) < 2:
        return _empty_result("numeric", "JS", "Insufficient variation for Jensen–Shannon")
    ref_hist = np.histogram(ref, bins=bins)[0].astype(float)
    prod_hist = np.histogram(prod, bins=bins)[0].astype(float)
    p = np.clip(ref_hist / len(ref), 1e-6, None)
    q = np.clip(prod_hist / len(prod), 1e-6, None)
    m = 0.5 * (p + q)
    divergence = float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))
    drifted = divergence >= threshold
    return {
        "type": "numeric", "test": "JS", "statistic": round(divergence, 4),
        "p_value": None, "drifted": drifted,
        "severity": "critical" if drifted else "info",
        "message": f"Jensen–Shannon={divergence:.4f} — {'drift detected' if drifted else 'no drift'}",
    }


def _chi2_test(ref_col: pd.Series, prod_col: pd.Series, threshold: float) -> dict[str, Any]:
    ref_counts = ref_col.value_counts()
    prod_counts = prod_col.value_counts()
    all_cats = ref_counts.index.union(prod_counts.index)
    if len(all_cats) == 0:
        return _empty_result("categorical", "Chi2", "No categories found for Chi2 test")
    # Plain dicts avoid Series.get() positional-fallback semantics (pandas FutureWarning).
    ref_map = dict(zip(ref_counts.index, ref_counts))
    prod_map = dict(zip(prod_counts.index, prod_counts))
    ref_vec = np.array([ref_map.get(c, 0) for c in all_cats], dtype=float) + 0.5
    prod_vec = np.array([prod_map.get(c, 0) for c in all_cats], dtype=float) + 0.5
    try:
        stat, p, _, _ = stats.chi2_contingency([ref_vec, prod_vec])
    except ValueError:
        return _empty_result("categorical", "Chi2", "Chi2 test could not be computed")
    drifted = bool(p < threshold)
    return {
        "type": "categorical", "test": "Chi2", "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4), "drifted": drifted,
        "severity": "critical" if drifted else "info",
        "message": f"Chi2 stat={stat:.4f}, p={p:.4f} — {'drift detected' if drifted else 'no drift'}",
    }
