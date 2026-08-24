"""Backend coverage tests for the default pandas data backend."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fitcheck.backends.pandas_backend import PandasBackend

try:
    import openpyxl  # noqa: F401
    # pandas>=2.1 requires openpyxl>=3.1.0 for Excel I/O
    HAVE_OPENPYXL = True
    _OPENPYXL_OK = pd.__version__ < "2.1.0" or tuple(
        int(x) for x in openpyxl.__version__.split(".")[:2]
    ) >= (3, 1)
except ImportError:
    HAVE_OPENPYXL = False
    _OPENPYXL_OK = False


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


@pytest.mark.skipif(
    not HAVE_OPENPYXL or not _OPENPYXL_OK,
    reason="openpyxl>=3.1.0 is required for Excel I/O with this pandas version",
)
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
