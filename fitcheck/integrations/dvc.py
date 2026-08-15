"""Lightweight DVC metrics writer (optional dependency)."""

from __future__ import annotations

from typing import Any


def log_to_dvc(result: dict[str, Any], stage: str = "validate", path: str = "fitcheck_metrics.yaml") -> bool:
    """Write FitCheck summary metrics as a DVC-compatible YAML metrics file.

    Returns True when PyYAML is installed and the file was written, False when
    the optional dependency is missing (no-op by design).
    """
    try:
        import yaml
    except ImportError:
        return False
    try:
        summary = result.get("summary", {})
        metrics = {
            stage: {
                "critical": summary.get("critical", 0),
                "warning": summary.get("warning", 0),
                "info": summary.get("info", 0),
                "n_issues": len(result.get("issues", [])),
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metrics, f)
        return True
    except OSError:  # pragma: no cover - filesystem edge case
        return False
