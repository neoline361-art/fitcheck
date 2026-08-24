"""Data backends: pandas (default) and optional polars/duckdb for fast loading."""

from __future__ import annotations

from typing import Any

from fitcheck.backends.base import DataBackend
from fitcheck.backends.pandas_backend import PandasBackend


def get_backend(name: str | None = None, df: Any | None = None) -> DataBackend:
    """Return the requested backend, defaulting to pandas.

    Auto-selects polars when ``df`` is already a polars DataFrame and no
    explicit name is given. An explicit ``polars`` or ``duckdb`` request raises
    ImportError when the corresponding package is not installed; an unknown
    backend name raises ValueError instead of silently falling back.
    """
    if name is None:
        if df is not None and type(df).__module__.startswith("polars"):
            from fitcheck.backends.polars_backend import PolarsBackend

            return PolarsBackend()
        return PandasBackend()
    if name == "pandas":
        return PandasBackend()
    if name == "polars":
        from fitcheck.backends.polars_backend import PolarsBackend

        return PolarsBackend()
    if name == "duckdb":
        from fitcheck.backends.duckdb_backend import DuckDBBackend

        return DuckDBBackend()
    raise ValueError(f"Unknown backend: {name!r}")


__all__ = ["DataBackend", "PandasBackend", "get_backend"]
