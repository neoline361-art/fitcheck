"""Abstract interface implemented by every FitCheck data backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataBackend(ABC):
    """Load tabular data from disk into a backend-native frame."""

    name: str = "base"

    @abstractmethod
    def read(self, path: str, **kwargs: Any) -> Any:
        """Read a CSV or Parquet file into a backend-native object."""

    def to_pandas(self, frame: Any) -> Any:
        """Convert a backend-native frame to pandas (checks are pandas-native)."""
        return frame.to_pandas()

    def is_supported(self, path: str) -> bool:
        """Whether this backend can read ``path``."""
        return path.endswith((".csv", ".parquet", ".json", ".arrow", ".feather", ".xlsx"))
