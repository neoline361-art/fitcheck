"""Optional MLflow / DVC integrations.

These modules import heavy tools lazily and no-op when they are absent, so
the core package never depends on MLflow or DVC.
"""

from __future__ import annotations

from fitcheck.integrations.dvc import log_to_dvc
from fitcheck.integrations.mlflow import log_to_mlflow

__all__ = ["log_to_mlflow", "log_to_dvc"]
