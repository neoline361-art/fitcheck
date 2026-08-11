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
) -> list[dict[str, Any]]:
    """Validate basic timestamp ordering, uniqueness, and missingness."""
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
    return issues
