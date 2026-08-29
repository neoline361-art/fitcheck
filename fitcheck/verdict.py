"""Verdict engine — PASS / WARN / BLOCK.

Consumes clusters from :mod:`fitcheck.decision` and a :class:`~fitcheck.policy.Policy`
to produce a single actionable verdict with a next-step recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fitcheck.decision import IssueCluster
from fitcheck.policy import Policy


@dataclass(frozen=True)
class Verdict:
    """Immutable verdict snapshot."""

    decision: str  # "PASS", "WARN", or "BLOCK"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    score: int  # aggregate score (sum of cluster scores)
    primary_cluster: IssueCluster | None = None
    next_action: str = ""
    all_clusters: list[IssueCluster] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary suitable for ``--json`` output."""
        result: dict[str, Any] = {
            "verdict": self.decision,
            "confidence": self.confidence,
            "score": self.score,
            "next_action": self.next_action,
            "primary_cluster": None,
            "clusters": [],
        }
        if self.primary_cluster is not None:
            result["primary_cluster"] = {
                "impact_area": self.primary_cluster.impact_area,
                "score": self.primary_cluster.score,
                "columns": self.primary_cluster.columns,
                "description": self.primary_cluster.description,
                "recommendation": self.primary_cluster.recommendation,
                "issue_count": len(self.primary_cluster.issues),
            }
        for cluster in self.all_clusters:
            result["clusters"].append({
                "impact_area": cluster.impact_area,
                "score": cluster.score,
                "columns": cluster.columns,
                "description": cluster.description,
                "recommendation": cluster.recommendation,
                "issue_count": len(cluster.issues),
            })
        return result


def compute_verdict(
    clusters: list[IssueCluster],
    policy: Policy | None = None,
) -> Verdict:
    """Compute a PASS / WARN / BLOCK verdict from clusters.

    Decision rules (from PRD):
      1. No issues → PASS
      2. Any cluster score >= ``block_score`` (default 8) → BLOCK
      3. Total score >= ``block_score`` → BLOCK
      4. Any critical-severity issue AND total score >= 5 → BLOCK
      5. Total score >= ``warn_score`` (default 4) → WARN
      6. Otherwise → PASS

    The ``primary_cluster`` is the cluster with the highest score.
    ``next_action`` is singular — the one thing to do next.
    """
    if policy is None:
        policy = Policy.default()

    if not clusters:
        return Verdict(
            decision="PASS",
            confidence="HIGH",
            score=0,
            next_action="Safe to proceed.",
        )

    total_score = sum(c.score for c in clusters)
    primary = clusters[0]  # already sorted desc by score

    has_critical = any(
        i.get("severity") == "critical"
        for c in clusters
        for i in c.issues
    )

    # Decision logic
    decision: str
    confidence: str

    if (
        primary.score >= policy.block_score
        or total_score >= policy.block_score
        or (has_critical and total_score >= 5)
    ):
        decision = "BLOCK"
        confidence = "HIGH" if has_critical else "MEDIUM"
    elif total_score >= policy.warn_score:
        decision = "WARN"
        confidence = "HIGH" if has_critical else "MEDIUM"
    else:
        decision = "PASS"
        confidence = "HIGH"

    # Next action — singular, specific
    if decision == "BLOCK":
        next_action = (
            f"STOP. {primary.description} — {primary.recommendation}."
        )
    elif decision == "WARN":
        next_action = (
            f"Proceed with caution. {primary.description} — "
            f"{primary.recommendation}."
        )
    else:
        next_action = "Safe to proceed. No material risks detected."

    return Verdict(
        decision=decision,
        confidence=confidence,
        score=total_score,
        primary_cluster=primary,
        next_action=next_action,
        all_clusters=clusters,
    )
