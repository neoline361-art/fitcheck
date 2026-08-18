"""Backend coverage tests for the default pandas data backend."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fitcheck.backends.pandas_backend import PandasBackend

try:
    import openpyxl  # noqa: F401
    HAVE_OPENPYXL = True
except ImportError:
    HAVE_OPENPYXL = False


@pytest.fixture
def backend() -> PandasBackend:
    return PandasBackend()


def test_read_csv_branch(backend: PandasBackend, tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    frame = backend.read(str(csv))
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["a", "b"]


def test_read_parquet_branch(backend: PandasBackend, tmp_path: Path) -> None:
    parquet = tmp_path / "data.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(parquet)
    frame = backend.read(str(parquet))
    assert list(frame.columns) == ["x"]


def test_read_json_branch(backend: PandasBackend, tmp_path: Path) -> None:
    json_file = tmp_path / "data.json"
    json_file.write_text('[{"k": 1}, {"k": 2}]')
    frame = backend.read(str(json_file))
    assert list(frame["k"]) == [1, 2]


@pytest.mark.skipif(not HAVE_OPENPYXL, reason="openpyxl is an optional dependency")
def test_read_excel_branch(backend: PandasBackend, tmp_path: Path) -> None:
    excel = tmp_path / "data.xlsx"
    pd.DataFrame({"y": [5, 6]}).to_excel(excel, index=False)
    frame = backend.read(str(excel))
    assert list(frame.columns) == ["y"]


def test_read_unsupported_raises(backend: PandasBackend) -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        backend.read("data.bin")


def test_to_pandas_is_identity(backend: PandasBackend) -> None:
    frame = pd.DataFrame({"z": [7, 8]})
    assert backend.to_pandas(frame) is frame
