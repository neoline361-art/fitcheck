"""FitCheck configuration dataclass.

Provides a typed, validated container for check thresholds that replaces
the plain ``dict[str, float]`` used internally by ``check()``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class FitCheckConfig:
    """Typed configuration for FitCheck thresholds.

    All values have sensible defaults matching the legacy hardcoded dict
    in ``check.py``.  Use :meth:`from_dict` to build from a user-supplied
    dictionary (e.g. loaded from ``fitcheck.yaml``).
    """

    missing_warning: float = 0.05
    missing_critical: float = 0.20
    duplicate_threshold: float = 0.05
    imbalance_threshold: float = 0.80
    outlier_threshold: float = 0.01
    high_cardinality_ratio: float = 0.95
    text_length_outlier_multiplier: float = 3.0

    # -- Conversion helpers ---------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        """Serialize to the plain-dict format expected by ``check()``."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitCheckConfig:
        """Build from a partial or full dictionary.

        Unknown keys are silently ignored so that forward-compatible
        policy files don't break older FitCheck versions.

        Raises:
            ValueError: If a value has the wrong type or the critical
                threshold is less than the warning threshold.
        """
        known = {f.name for f in fields(cls)}
        filtered: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            filtered[key] = value
        try:
            instance = cls(**filtered)
        except TypeError as exc:
            raise ValueError(f"Invalid config value: {exc}") from exc
        try:
            if instance.missing_critical < instance.missing_warning:
                raise ValueError(
                    "missing_critical must be >= missing_warning"
                )
        except TypeError:
            raise ValueError(
                "Config values must be numeric; "
                f"got missing_critical={instance.missing_critical!r}, "
                f"missing_warning={instance.missing_warning!r}"
            )
        return instance

    def merge(self, overrides: dict[str, Any]) -> FitCheckConfig:
        """Return a new config with *overrides* applied on top."""
        base = self.to_dict()
        base.update(overrides)
        return FitCheckConfig.from_dict(base)
