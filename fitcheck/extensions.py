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
