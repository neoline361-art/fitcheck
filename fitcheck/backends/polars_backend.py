"""Optional polars backend.

Checks run on pandas (the check engine is pandas-native); polars accelerates
the load step for large CSV/Parquet files. Full polars-native checks are a
future optimisation, not a Week 2 requirement.
"""

from __future__ import annotations

from typing import Any

from fitcheck.backends.base import DataBackend


class PolarsBackend(DataBackend):
    """Read with polars and convert to pandas for the check engine."""

    name = "polars"

    def __init__(self) -> None:
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "The polars backend requires the 'polars' package; install it with "
                "pip install data-fitcheck[polars]"
            ) from exc
        self._pl = pl

    def read(self, path: str, **kwargs: Any) -> Any:
        if not self.is_supported(path):
            raise ValueError(f"Unsupported file type: {path}")
        if path.endswith(".parquet"):
            return self._pl.read_parquet(path, **kwargs)
        return self._pl.read_csv(path, **kwargs)

    def to_pandas(self, frame: Any) -> Any:
        return frame.to_pandas()
