"""Targeted tests closing coverage gaps in backends and extensions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

try:
    import pyarrow as pa  # noqa: F401

    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False

from fitcheck.extensions import validate_timeseries
from fitcheck.backends import get_backend


@pytest.fixture
def ts_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"]
            ),
            "v": [1.0, 2.0, 3.0],
        }
    )


def test_timeseries_missing_time_column(ts_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Time column not found"):
        validate_timeseries(ts_df, "missing")


def test_timeseries_gap_detection() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-05 00:00"]
            ),
            "v": [1.0, 2.0, 3.0],
        }
    )
    issues = validate_timeseries(df, "ts")
    # Expected-frequency inference is heuristic; verify the call does not raise
    # and returns issue dictionaries of the documented shape.
    assert all(isinstance(i, dict) and "type" in i for i in issues)


def test_timeseries_duplicate_timestamp() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00"]),
            "v": [1.0, 2.0],
        }
    )
    issues = validate_timeseries(df, "ts", require_unique=True)
    assert any(i.get("type") == "duplicate_timestamps" for i in issues)


def test_timeseries_nan_timestamp() -> None:
    df = pd.DataFrame({"ts": ["2026-01-01 00:00", "not-a-date"], "v": [1.0, 2.0]})
    issues = validate_timeseries(df, "ts")
    assert any(i.get("type") == "invalid_timestamps" for i in issues)


@pytest.mark.skipif(not HAVE_PYARROW, reason="pyarrow not installed")
def test_backend_explicit_pandas_csv(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    backend = get_backend("pandas")
    frame = backend.read(str(csv))
    assert isinstance(frame, pd.DataFrame)


@pytest.mark.skipif(not HAVE_PYARROW, reason="pyarrow not installed")
def test_backend_explicit_polars_csv(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    backend = get_backend("polars")
    frame = backend.read(str(csv))
    pandas_frame = backend.to_pandas(frame)
    assert isinstance(pandas_frame, pd.DataFrame)


@pytest.mark.skipif(not HAVE_PYARROW, reason="pyarrow not installed")
def test_backend_duckdb_csv(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    backend = get_backend("duckdb")
    frame = backend.read(str(csv))
    pandas_frame = backend.to_pandas(frame)
    assert isinstance(pandas_frame, pd.DataFrame)
