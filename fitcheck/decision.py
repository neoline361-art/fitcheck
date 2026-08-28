"""Issue clustering and impact scoring for the decision engine.

Groups raw issues from ``check()`` into logical clusters and assigns
an impact score (1–10) based on issue type, severity, and affected-column
concentration.  **This module does NOT modify check.py** — it only
consumes the issue dicts that ``check()`` produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Impact inference map ────────────────────────────────────────────────
# Maps each issue type to the downstream component it most affects.
_IMPACT_MAP: dict[str, str] = {
    "missing_values": "training",
    "duplicate_rows": "metrics",
    "constant_column": "training",
    "class_imbalance": "training",
    "outliers": "inference",
    "high_cardinality": "training",
    "text_encoding": "pipeline",
    "text_length_outliers": "training",
    "schema_change": "pipeline",
    "distribution_shift": "inference",
}

_SEVERITY_SCORE: dict[str, int] = {
    "critical": 4,
    "warning": 2,
    "info": 1,
}


@dataclass
class IssueCluster:
    """A group of related issues sharing a root cause."""

    impact_area: str  # e.g. "training", "inference", "pipeline"
    issues: list[dict[str, Any]] = field(default_factory=list)
    score: int = 0  # 1–10

    @property
    def columns(self) -> list[str]:
        """Unique columns affected by issues in this cluster."""
        seen: list[str] = []
        for issue in self.issues:
            col = issue.get("column", "")
            if col not in seen:
                seen.append(col)
        return seen

    @property
    def description(self) -> str:
        """Human-readable summary of the cluster."""
        if not self.issues:
            return "No issues"
        types = sorted({i.get("type", "unknown") for i in self.issues})
        return f"Cluster: {', '.join(types)} ({self.impact_area})"

    @property
    def recommendation(self) -> str:
        """Suggested next action for this cluster."""
        if self.score >= 8:
            return "BLOCK — fix before proceeding"
        if self.score >= 5:
            return "Investigate before next step"
        return "Monitor — low immediate risk"


def cluster_issues(issues: list[dict[str, Any]]) -> list[IssueCluster]:
    """Group issues by their inferred impact area.

    Issues of the same ``type`` that share the same ``impact_area`` are
    placed into a single :class:`IssueCluster`, sorted by descending score.

    Returns a list of clusters, highest impact first.
    """
    if not issues:
        return []

    # Group by impact_area
    groups: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        issue_type = issue.get("type", "unknown")
        area = _IMPACT_MAP.get(issue_type, "unknown")
        groups.setdefault(area, []).extend([issue])

    # Within each area, sub-group by type so each cluster is one issue type
    clusters: list[IssueCluster] = []
    for area, area_issues in groups.items():
        type_groups: dict[str, list[dict[str, Any]]] = {}
        for issue in area_issues:
            t = issue.get("type", "unknown")
            type_groups.setdefault(t, []).append(issue)

        for _type, group_issues in type_groups.items():
            cluster = IssueCluster(
                impact_area=area,
                issues=group_issues,
            )
            cluster.score = _compute_score(cluster)
            clusters.append(cluster)

    clusters.sort(key=lambda c: c.score, reverse=True)
    return clusters


def _compute_score(cluster: IssueCluster) -> int:
    """Compute a 1–10 impact score for a cluster.

    Heuristic:
      - Start with severity sum (critical=4, warning=2, info=1).
      - If multiple columns affected, +1 bonus.
      - If 5+ issues, +1 bonus.
      - Cap at 10.
    """
    base = sum(
        _SEVERITY_SCORE.get(i.get("severity", "info"), 1)
        for i in cluster.issues
    )
    cols = cluster.columns
    if len(cols) > 1:
        base += 1
    if len(cluster.issues) >= 5:
        base += 1
    return min(base, 10)
