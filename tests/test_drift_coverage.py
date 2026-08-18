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


def test_drift_dtype_kind_classification(tmp_path: Path) -> None:
    ref = tmp_path / "ref.csv"
    prod = tmp_path / "prod.csv"
    # CSV round-trip preserves int/float dtype kinds (i vs f differ), which
    # exercises the bool/datetime/numeric dtype-kind branches of the schema
    # drift classifier.
    ref_frame = pd.DataFrame({"n": [1, 2], "s": ["a", "b"]})
    prod_frame = pd.DataFrame({"n": [1.5, 2.5], "s": ["a", "b"]})
    _write(ref_frame, ref)
    _write(prod_frame, prod)
    results = fitcheck.detect_drift(str(ref), str(prod))
    messages = " ".join(str(r.get("message", "")) for r in results)
    assert "dtype changed" in messages


def test_drift_parquet_loading_branch(tmp_path: Path) -> None:
    ref = tmp_path / "ref.parquet"
    prod = tmp_path / "prod.parquet"
    frame = pd.DataFrame({"a": [1, 2, 3]})
    frame.to_parquet(ref)
    frame.to_parquet(prod)
    results = fitcheck.detect_drift(str(ref), str(prod))
    assert len(results) >= 1
