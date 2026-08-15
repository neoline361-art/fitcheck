"""Default pandas-based data backend."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fitcheck.backends.base import DataBackend


class PandasBackend(DataBackend):
    """Read data directly with pandas; frames are already pandas-native."""

    name = "pandas"

    def read(self, path: str, **kwargs: Any) -> pd.DataFrame:
        if not self.is_supported(path):
            raise ValueError(f"Unsupported file type: {path}")
        if path.endswith(".parquet"):
            return pd.read_parquet(path, **kwargs)
        if path.endswith((".json", ".jsonl")):
            return pd.read_json(path, **kwargs)
        if path.endswith((".xlsx", ".xls")):
            return pd.read_excel(path, **kwargs)
        return pd.read_csv(path, **kwargs)

    def to_pandas(self, frame: Any) -> pd.DataFrame:
        return frame
