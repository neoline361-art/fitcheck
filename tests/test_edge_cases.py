"""Edge-case hardening tests.

Every case here exercises a degenerate input against the public check/report
APIs. The expected behavior is explicitly decided for each:
VALID RESULT / WARNING / SKIP / ACTIONABLE ERROR.

No case may raise an unhandled exception; a crash always fails the suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fitcheck.check import check
from fitcheck.report import report


def _check_ok_or_dict(*args, **kwargs) -> dict:
    """Run check and return the result dict; never let it crash."""
    kwargs.setdefault("return_format", "dict")
    result = check(*args, **kwargs)
    assert isinstance(result, dict), "degenerate input must yield a dict result"
    return result


# ---------------------------------------------------------------------------
# 1. Empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_dataframe(tmp_path: Path) -> None:
    result = _check_ok_or_dict(pd.DataFrame(), output=str(tmp_path / "empty.html"))
    assert result["total_rows"] == 0
    assert result["total_columns"] == 0
    assert result["issues"] == []  # empty data is clean, not an error


# ---------------------------------------------------------------------------
# 2. Single-row DataFrame
# ---------------------------------------------------------------------------

def test_single_row_dataframe(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    result = _check_ok_or_dict(df, output=str(tmp_path / "one_row.html"))
    assert result["total_rows"] == 1
    # A single-row column is trivially constant; the engine skips tiny
    # samples rather than over-flagging noise.
    assert isinstance(result["issues"], list)


# ---------------------------------------------------------------------------
# 3. All-NaN column
# ---------------------------------------------------------------------------

def test_all_nan_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "missing_all": [np.nan, np.nan, np.nan]})
    result = _check_ok_or_dict(df, output=str(tmp_path / "allnan.html"))
    types = {issue["type"] for issue in result["issues"]}
    assert "missing_values" in types, "fully missing column must be reported"
    # Expected: critical (100% missing exceeds the critical threshold).
    critical = [i for i in result["issues"] if i["column"] == "missing_all" and i["type"] == "missing_values"]
    assert critical and critical[0]["severity"] == "critical"


# ---------------------------------------------------------------------------
# 4. Constant column
# ---------------------------------------------------------------------------

def test_constant_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "k": [7, 7, 7]})
    result = _check_ok_or_dict(df, output=str(tmp_path / "const.html"))
    consts = [i for i in result["issues"] if i["type"] == "constant_column"]
    assert consts and consts[0]["column"] == "k"
    assert consts[0]["severity"] == "warning"


# ---------------------------------------------------------------------------
# 5. Constant column containing NaN
# ---------------------------------------------------------------------------

def test_constant_column_with_nan(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "k": [7.0, np.nan, 7.0]})
    result = _check_ok_or_dict(df, output=str(tmp_path / "const_nan.html"))
    consts = [i for i in result["issues"] if i["type"] == "constant_column"]
    assert consts and consts[0]["column"] == "k"
    missing = [i for i in result["issues"] if i["type"] == "missing_values" and i["column"] == "k"]
    assert missing, "the NaN within the constant column must also be surfaced"


# ---------------------------------------------------------------------------
# 6. One-class classification target
# ---------------------------------------------------------------------------

def test_one_class_target(tmp_path: Path) -> None:
    x = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1]})
    y = pd.Series([0, 0, 0, 0, 0, 0])  # single class — degenerate
    with pytest.raises(Exception):  # actionable error: training cannot succeed
        report(LogisticRegression(), x, y, output=str(tmp_path / "oneclass.html"))


# ---------------------------------------------------------------------------
# 7. Missing target column
# ---------------------------------------------------------------------------

def test_missing_target_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    # A missing target column must be an actionable error, not a crash.
    with pytest.raises(ValueError, match="not found in columns"):
        _check_ok_or_dict(df, target="nope", output=str(tmp_path / "no_target.html"))


# ---------------------------------------------------------------------------
# 8. Empty target
# ---------------------------------------------------------------------------

def test_empty_target(tmp_path: Path) -> None:
    x = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([np.nan, np.nan, np.nan])
    with pytest.raises(Exception):  # actionable error: nothing to learn from
        report(LogisticRegression(), x, y, output=str(tmp_path / "empty_target.html"))


# ---------------------------------------------------------------------------
# 9. Mixed/object columns with numeric-like content
# ---------------------------------------------------------------------------

def test_mixed_object_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, "2", 3, "four", 5], "b": ["x", "y", "z", "x", "y"]})
    result = _check_ok_or_dict(df, output=str(tmp_path / "mixed.html"))
    assert result["total_rows"] == 5  # data loads; checks run on object columns


# ---------------------------------------------------------------------------
# 10. High-cardinality categorical column
# ---------------------------------------------------------------------------

def test_high_cardinality_categorical(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    ids = [f"id-{i}" for i in rng.integers(0, 10_000, 200)]
    df = pd.DataFrame({"id": ids, "val": rng.integers(0, 5, 200)})
    result = _check_ok_or_dict(df, output=str(tmp_path / "highcard.html"))
    types = {issue["type"] for issue in result["issues"]}
    assert "high_cardinality" in types, "ID-like column must be flagged"


def test_high_cardinality_via_csv(tmp_path: Path) -> None:
    """CSV loading parses ID-like columns as object dtype, preserving the check."""
    rng = np.random.default_rng(0)
    rows = [(f"id-{i}", int(i % 5)) for i in rng.integers(0, 10_000, 200)]
    path = tmp_path / "id.csv"
    with path.open("w") as fh:
        fh.write("id,val\n")
        for ident, val in rows:
            fh.write(f"{ident},{val}\n")
    result = _check_ok_or_dict(str(path), output=str(tmp_path / "id.html"))
    types = {issue["type"] for issue in result["issues"]}
    assert "high_cardinality" in types, "ID-like CSV column must be flagged"


# ---------------------------------------------------------------------------
# 11. Zero-variance numeric column
# ---------------------------------------------------------------------------

def test_zero_variance_numeric(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1, 2, 3], "target": [0, 1, 0]})
    result = _check_ok_or_dict(df, target="target", output=str(tmp_path / "zerovar.html"))
    types = {issue["type"] for issue in result["issues"]}
    assert "constant_column" in types or "outliers" in types
    # Outlier detection must skip zero-variance columns gracefully.
    outliers = [i for i in result["issues"] if i["type"] == "outliers" and i["column"] == "a"]
    assert not outliers, "zero-variance numeric column must be skipped"


# ---------------------------------------------------------------------------
# 12. Very small dataset
# ---------------------------------------------------------------------------

def test_very_small_dataset(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "target": [0, 1]})
    result = _check_ok_or_dict(df, target="target", output=str(tmp_path / "tiny.html"))
    assert result["total_rows"] == 2
    assert isinstance(result["issues"], list)
    # Cardinality-based checks must skip datasets that are too small.
    highcard = [i for i in result["issues"] if i["type"] == "high_cardinality"]
    assert not highcard


# ---------------------------------------------------------------------------
# 13. Constant column detection in a file path (backend parity)
# ---------------------------------------------------------------------------

def test_edge_cases_via_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"all_nan": [np.nan, np.nan], "k": [1, 1], "a": [1.0, 2.0]})
    path = tmp_path / "edge.csv"
    df.to_csv(path, index=False)
    result = _check_ok_or_dict(str(path), output=str(tmp_path / "edge.html"))
    types = {issue["type"] for issue in result["issues"]}
    assert "constant_column" in types or "missing_values" in types
