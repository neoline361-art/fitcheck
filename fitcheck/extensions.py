from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

CheckPlugin = Callable[[pd.DataFrame], list[dict[str, Any]]]


def run_plugins(data: pd.DataFrame, plugins: list[CheckPlugin] | None = None) -> list[dict[str, Any]]:
    """Run opt-in custom checks without modifying the input DataFrame."""
    issues: list[dict[str, Any]] = []
    for plugin in plugins or []:
        result = plugin(data.copy())
        if not isinstance(result, list):
            raise TypeError(f"Plugin {plugin!r} must return a list of issue dictionaries")
        issues.extend(result)
    return issues


def validate_timeseries(
    data: pd.DataFrame,
    time_column: str,
    *,
    require_monotonic: bool = True,
    require_unique: bool = True,
    gap_multiplier: float = 3.0,
) -> list[dict[str, Any]]:
    """Validate basic timestamp ordering, uniqueness, missingness, and frequency gaps."""
    if time_column not in data.columns:
        raise KeyError(f"Time column not found: {time_column}")
    values = pd.to_datetime(data[time_column], errors="coerce")
    issues: list[dict[str, Any]] = []
    missing = int(values.isna().sum())
    if missing:
        issues.append({"column": time_column, "type": "invalid_timestamps", "severity": "critical", "message": f"{missing} timestamp values could not be parsed", "suggestion": "Parse timestamps before validation and handle invalid rows explicitly"})
    valid = values.dropna()
    if require_unique and valid.duplicated().any():
        issues.append({"column": time_column, "type": "duplicate_timestamps", "severity": "warning", "message": f"{int(valid.duplicated().sum())} duplicate timestamps found", "suggestion": "Aggregate or disambiguate duplicate timestamps"})
    if require_monotonic and not valid.is_monotonic_increasing:
        issues.append({"column": time_column, "type": "non_monotonic_time", "severity": "warning", "message": "Timestamps are not sorted in ascending order", "suggestion": f'data.sort_values("{time_column}") explicitly before time-series operations'})
    issues.extend(_detect_ts_gaps(valid, time_column, gap_multiplier))
    return issues


def detect_seasonality(series: pd.Series, period: int | None = None) -> dict[str, Any] | None:
    """Return an info issue when the series shows a repeatable seasonal pattern.

    Uses autocorrelation at a candidate lag (upgrade path: STL decomposition via
    ``statsmodels.tsa.seasonal.STL`` once statsmodels becomes a dependency).
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 30:
        return None
    lag = period or _infer_seasonal_lag(values)
    if lag is None or lag < 2 or lag >= len(values) // 3:
        return None
    autocorr = float(values.autocorr(lag=lag))
    if autocorr >= 0.5:
        return {
            "column": getattr(series, "name", ""),
            "type": "timeseries_seasonality",
            "severity": "info",
            "message": f"Possible {lag}-step seasonality detected (autocorrelation {autocorr:.2f})",
            "suggestion": "Confirm with domain knowledge, then model the seasonal component explicitly",
        }
    return None


def _infer_seasonal_lag(values: pd.Series) -> int | None:
    """Pick the lag with the strongest positive autocorrelation among common periods."""
    best_lag, best_score = None, 0.5
    for lag in (7, 12, 24, 30):  # daily-weekly, monthly, hourly-daily, daily-monthly
        if lag >= len(values) // 3:
            continue
        try:
            score = float(values.autocorr(lag=lag))
        except ValueError:
            continue
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag


def _detect_ts_gaps(values: pd.Series, time_column: str, gap_multiplier: float) -> list[dict[str, Any]]:
    """Flag time gaps that exceed the expected sampling frequency."""
    diffs = values.dropna().sort_values().diff().dropna()
    if len(diffs) < 2:
        return []
    median = diffs.median()
    if median <= pd.Timedelta(0) or pd.isna(median):
        return []
    big_gaps = diffs[diffs > gap_multiplier * median]
    if not len(big_gaps):
        return []
    count = int(len(big_gaps))
    largest = big_gaps.max()
    return [
        {
            "column": time_column,
            "type": "time_series_gaps",
            "severity": "critical",
            "message": f"{count} gaps exceeding {gap_multiplier:.0f}x the median interval (largest: {largest})",
            "suggestion": "Resample to the expected frequency or investigate missing periods",
        }
    ]
