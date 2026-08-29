"""Policy engine for FitCheck decision mode.

Loads severity thresholds from a ``fitcheck.yaml`` file (auto-detected in
CWD or overridden via ``--policy``).  The policy defines per-issue-type
severity overrides and the fail thresholds for the verdict engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # PyYAML is already a project dependency


@dataclass(frozen=True)
class Policy:
    """Immutable policy snapshot loaded from YAML or defaults."""

    block_score: int = 8
    warn_score: int = 4
    issue_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> Policy:
        """Return the hardcoded default policy."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        """Build from a parsed YAML dictionary.

        Raises:
            ValueError: If thresholds are inconsistent.
        """
        ft = data.get("fail_thresholds", {})
        block_score = ft.get("block_score", 8)
        warn_score = ft.get("warn_score", 4)
        if block_score < warn_score:
            raise ValueError(
                "block_score must be >= warn_score"
            )
        overrides = data.get("issue_overrides", {})
        return cls(
            block_score=block_score,
            warn_score=warn_score,
            issue_overrides=overrides,
        )


def load_policy(path: str | Path | None = None) -> Policy:
    """Load a policy from *path*, or auto-detect ``fitcheck.yaml`` in CWD.

    If no file is found, the default policy is returned.

    Raises:
        ValueError: If the YAML content is invalid or thresholds conflict.
        FileNotFoundError: If an explicit *path* does not exist.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {p}")
        return _parse_yaml(p)

    # Auto-detect in CWD
    candidates = [
        Path("fitcheck.yaml"),
        Path("fitcheck.yml"),
        Path(".fitcheck.yaml"),
        Path(".fitcheck.yml"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return _parse_yaml(candidate)

    return Policy.default()


def _parse_yaml(path: Path) -> Policy:
    """Parse and validate a YAML policy file."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Policy file must contain a YAML mapping: {path}")
    return Policy.from_dict(raw)
