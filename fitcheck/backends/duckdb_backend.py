"""Optional duckdb backend.

Checks run on pandas (the check engine is pandas-native); duckdb accelerates
the load step for large CSV/Parquet files, including files larger than RAM.
"""

from __future__ import annotations

from typing import Any

from fitcheck.backends.base import DataBackend


class DuckDBBackend(DataBackend):
    """Read with duckdb and convert to pandas for the check engine."""

    name = "duckdb"

    def __init__(self) -> None:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "The duckdb backend requires the 'duckdb' package; install it with "
                "pip install data-fitcheck[duckdb]"
            ) from exc
        self._duckdb = duckdb

    def read(self, path: str, **kwargs: Any) -> Any:
        if not self.is_supported(path):
            raise ValueError(f"Unsupported file type: {path}")
        if path.endswith(".parquet"):
            return self._duckdb.read_parquet(path, **kwargs)
        return self._duckdb.read_csv(path, **kwargs)

    def to_pandas(self, frame: Any) -> Any:
        return frame.df()
