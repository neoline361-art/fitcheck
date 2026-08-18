"""Targeted tests closing coverage gaps in drift.py error and edge branches."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import fitcheck

# These branches are exercised only by degenerate inputs that the standard
# drift suite does not produce, so they live in a dedicated file.


def test_drift_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        fitcheck.detect_drift("a.csv", "b.csv", threshold=1.5)


def test_drift_unsupported_method() -> None:
    with pytest.raises(ValueError, match="Unsupported drift method"):
        fitcheck.detect_drift("a.csv", "b.csv", method="bogus")


def test_drift_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        fitcheck.detect_drift("/nonexistent/ref.csv", "/nonexistent/prod.csv")


def _write(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        frame.to_parquet(path)
    else:
        frame.to_csv(path, index=False)


def test_drift_schema_extra_column_in_production(tmp_path: Path) -> None:
    ref = tmp_path / "ref.csv"
    prod = tmp_path / "prod.csv"
    _write(pd.DataFrame({"a": [1, 2]}), ref)
    _write(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), prod)
    results = fitcheck.detect_drift(str(ref), str(prod))
    assert any("missing in reference" in str(r.get("message", "")) for r in results)


def test_drift_insufficient_numeric_data_ks(tmp_path: Path) -> None:
    ref = tmp_path / "ref.csv"
    prod = tmp_path / "prod.csv"
    _write(pd.DataFrame({"a": ["x", "y"]}), ref)  # non-numeric column
    _write(pd.DataFrame({"a": ["x", "y"]}), prod)
    results = fitcheck.detect_drift(str(ref), str(prod), method="ks")
    assert any(r.get("test") == "KS" for r in results)


def test_drift_insufficient_variation_psi(tmp_path: Path) -> None:
    ref = tmp_path / "ref.csv"
    prod = tmp_path / "prod.csv"
    _write(pd.DataFrame({"a": [1, 1, 1]}), ref)  # zero variation
    _write(pd.DataFrame({"a": [2, 2, 2]}), prod)
    results = fitcheck.detect_drift(str(ref), str(prod), method="psi")
    assert any(r.get("test") == "PSI" for r in results)


def test_drift_insufficient_data_wasserstein(tmp_path: Path) -> None:
    ref = tmp_path / "ref.csv"
    prod = tmp_path / "prod.csv"
    _write(pd.DataFrame({"a": ["x", "y"]}), ref)
    _write(pd.DataFrame({"a": ["x", "y"]}), prod)
    results = fitcheck.detect_drift(str(ref), str(prod), method="wasserstein")
    assert any(r.get("test") == "Wasserstein" for r in results)


def test_drift_insufficient_variation_js(tmp_path: Path) -> None:
    ref = tmp_path / "ref.csv"
    prod = tmp_path / "prod.csv"
    _write(pd.DataFrame({"a": [5, 5, 5]}), ref)
    _write(pd.DataFrame({"a": [6, 6, 6]}), prod)
    results = fitcheck.detect_drift(str(ref), str(prod), method="js")
    assert any(r.get("test") == "JS" for r in results)


def test_drift_no_categories_chi2(tmp_path: Path) -> None:
    ref = tmp_path / "ref.csv"
    prod = tmp_path / "prod.csv"
    _write(pd.DataFrame({"a": [1, 2]}), ref)  # numeric, not categorical
    _write(pd.DataFrame({"a": [3, 4]}), prod)
    results = fitcheck.detect_drift(str(ref), str(prod), method="chi2")
    assert any(r.get("test") == "Chi2" for r in results)


def test_drift_dtype_kind_classification() -> None:
    # The schema classifier groups int/uint/float as one numeric family; it
    # only fires on bool, datetime, or object kind changes. Exercise the
    # dtype-kind branches directly and through a real schema change.
    from fitcheck.drift import _dtype_kind

    assert _dtype_kind(pd.api.types.pandas_dtype("bool")) == "bool"
    assert _dtype_kind(pd.api.types.pandas_dtype("datetime64[ns]")) == "datetime"
    assert _dtype_kind(pd.api.types.pandas_dtype("int64")) == "numeric"
    assert _dtype_kind(pd.api.types.pandas_dtype("object")) == "object"
    ref = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    prod = pd.DataFrame({"ts": ["2026-01-01", "2026-01-02"]})
    from fitcheck.drift import _schema_drift

    schema = _schema_drift(ref, prod)
    assert any("dtype changed" in str(s.get("message", "")) for s in schema)


def test_drift_parquet_loading_branch(tmp_path: Path) -> None:
    ref = tmp_path / "ref.parquet"
    prod = tmp_path / "prod.parquet"
    frame = pd.DataFrame({"a": [1, 2, 3]})
    frame.to_parquet(ref)
    frame.to_parquet(prod)
    results = fitcheck.detect_drift(str(ref), str(prod))
    assert len(results) >= 1
