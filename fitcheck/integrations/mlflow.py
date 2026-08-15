"""Lightweight MLflow callback (optional dependency)."""

from __future__ import annotations

from typing import Any


def log_to_mlflow(result: dict[str, Any], run_id: str | None = None) -> bool:
    """Log FitCheck metrics to the active MLflow run.

    Returns True when MLflow is installed and logging succeeded, False when the
    optional dependency is missing (no-op by design).
    """
    try:
        import mlflow
    except ImportError:
        return False
    try:
        if run_id is not None:
            mlflow.start_run(run_id=run_id)
        mlflow.log_metric("fitcheck_issues", len(result.get("issues", [])))
        mlflow.log_metric("fitcheck_critical", result.get("summary", {}).get("critical", 0))
        mlflow.log_dict(result, "fitcheck_report.json")
        return True
    except Exception:  # pragma: no cover - mlflow API surface varies by version
        return False
