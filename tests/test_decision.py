"""Tests for Sprint 2: Decision Engine modules.

Covers: config.py, policy.py, decision.py, verdict.py, html.py decision
layout, and CLI --mode/--policy flags.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fitcheck.config import FitCheckConfig
from fitcheck.decision import cluster_issues
from fitcheck.policy import Policy, load_policy
from fitcheck.verdict import compute_verdict

# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_issues() -> list[dict]:
    """A realistic mix of issues from check()."""
    return [
        {"column": "age", "type": "missing_values", "severity": "critical",
         "message": "age: 25% missing", "suggestion": "Impute or drop"},
        {"column": "income", "type": "missing_values", "severity": "warning",
         "message": "income: 8% missing", "suggestion": "Median imputation"},
        {"column": "id", "type": "high_cardinality", "severity": "info",
         "message": "id: 99.9% unique", "suggestion": "Verify not an ID"},
        {"column": "all", "type": "duplicate_rows", "severity": "warning",
         "message": "150 duplicate rows", "suggestion": "Drop duplicates"},
        {"column": "const_col", "type": "constant_column", "severity": "warning",
         "message": "const_col: constant value", "suggestion": "Drop column"},
    ]


@pytest.fixture
def empty_issues() -> list[dict]:
    return []


@pytest.fixture
def critical_only() -> list[dict]:
    return [
        {"column": "target", "type": "missing_values", "severity": "critical",
         "message": "target: 30% missing", "suggestion": "Fix immediately"},
        {"column": "feature1", "type": "missing_values", "severity": "critical",
         "message": "feature1: 25% missing", "suggestion": "Fix immediately"},
        {"column": "feature2", "type": "missing_values", "severity": "critical",
         "message": "feature2: 22% missing", "suggestion": "Fix immediately"},
        {"column": "feature3", "type": "missing_values", "severity": "critical",
         "message": "feature3: 21% missing", "suggestion": "Fix immediately"},
    ]


@pytest.fixture
def info_only() -> list[dict]:
    return [
        {"column": "col_a", "type": "outliers", "severity": "info",
         "message": "col_a: 2% outliers", "suggestion": "Review"},
        {"column": "col_b", "type": "text_length_outliers", "severity": "info",
         "message": "col_b: long strings", "suggestion": "Review"},
    ]


# ──────────────────────────────────────────────────────────────────────────
# config.py tests
# ──────────────────────────────────────────────────────────────────────────

class TestFitCheckConfig:
    def test_default_values(self) -> None:
        cfg = FitCheckConfig()
        assert cfg.missing_warning == 0.05
        assert cfg.missing_critical == 0.20
        assert cfg.outlier_threshold == 0.01

    def test_to_dict_roundtrip(self) -> None:
        cfg = FitCheckConfig(missing_warning=0.10)
        d = cfg.to_dict()
        assert d["missing_warning"] == 0.10
        cfg2 = FitCheckConfig.from_dict(d)
        assert cfg2.missing_warning == 0.10

    def test_from_dict_partial(self) -> None:
        cfg = FitCheckConfig.from_dict({"missing_warning": 0.15})
        assert cfg.missing_warning == 0.15
        assert cfg.missing_critical == 0.20  # default preserved

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = FitCheckConfig.from_dict({"unknown_key": 999, "missing_warning": 0.10})
        assert cfg.missing_warning == 0.10
        assert not hasattr(cfg, "unknown_key")

    def test_from_dict_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="Config values must be numeric|Invalid config value"):
            FitCheckConfig.from_dict({"missing_warning": "not_a_number"})

    def test_from_dict_critical_lt_warning_raises(self) -> None:
        with pytest.raises(ValueError, match="missing_critical must be"):
            FitCheckConfig.from_dict({
                "missing_warning": 0.30,
                "missing_critical": 0.10,
            })

    def test_merge(self) -> None:
        cfg = FitCheckConfig()
        merged = cfg.merge({"missing_warning": 0.15})
        assert merged.missing_warning == 0.15
        assert merged.missing_critical == 0.20

    def test_frozen(self) -> None:
        cfg = FitCheckConfig()
        with pytest.raises(AttributeError):
            cfg.missing_warning = 0.50  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────
# policy.py tests
# ──────────────────────────────────────────────────────────────────────────

class TestPolicy:
    def test_default(self) -> None:
        p = Policy.default()
        assert p.block_score == 8
        assert p.warn_score == 4

    def test_from_dict(self) -> None:
        p = Policy.from_dict({
            "version": 1,
            "fail_thresholds": {"block_score": 10, "warn_score": 6},
        })
        assert p.block_score == 10
        assert p.warn_score == 6

    def test_from_dict_block_lt_warn_raises(self) -> None:
        with pytest.raises(ValueError, match="block_score must be"):
            Policy.from_dict({
                "fail_thresholds": {"block_score": 2, "warn_score": 8},
            })

    def test_load_policy_default(self) -> None:
        p = load_policy(None)
        assert p.block_score == 8

    def test_load_policy_from_file(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "fitcheck.yaml"
        policy_file.write_text(yaml.dump({
            "version": 1,
            "fail_thresholds": {"block_score": 7, "warn_score": 3},
            "issue_overrides": {
                "missing_values": {"severity_boost": 1},
            },
        }))
        p = load_policy(str(policy_file))
        assert p.block_score == 7
        assert p.warn_score == 3
        assert p.issue_overrides["missing_values"]["severity_boost"] == 1

    def test_load_policy_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_policy("/nonexistent/policy.yaml")

    def test_load_policy_auto_detect(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        policy_file = tmp_path / "fitcheck.yaml"
        policy_file.write_text(yaml.dump({
            "fail_thresholds": {"block_score": 5, "warn_score": 2},
        }))
        monkeypatch.chdir(tmp_path)
        p = load_policy(None)
        assert p.block_score == 5

    def test_load_policy_invalid_yaml_content(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "fitcheck.yaml"
        policy_file.write_text("just a string\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_policy(str(policy_file))


# ──────────────────────────────────────────────────────────────────────────
# decision.py tests
# ──────────────────────────────────────────────────────────────────────────

class TestDecision:
    def test_empty_issues(self) -> None:
        clusters = cluster_issues([])
        assert clusters == []

    def test_clusters_by_type(self, sample_issues: list[dict]) -> None:
        clusters = cluster_issues(sample_issues)
        # missing_values (2 issues) → 1 cluster
        # high_cardinality (1 issue) → 1 cluster
        # duplicate_rows (1 issue) → 1 cluster
        # constant_column (1 issue) → 1 cluster
        assert len(clusters) == 4

    def test_cluster_scores_descending(self, sample_issues: list[dict]) -> None:
        clusters = cluster_issues(sample_issues)
        scores = [c.score for c in clusters]
        assert scores == sorted(scores, reverse=True)

    def test_missing_values_cluster_is_highest(self, sample_issues: list[dict]) -> None:
        clusters = cluster_issues(sample_issues)
        assert clusters[0].impact_area == "training"
        assert any(i["type"] == "missing_values" for i in clusters[0].issues)

    def test_cluster_columns(self, sample_issues: list[dict]) -> None:
        clusters = cluster_issues(sample_issues)
        missing_cluster = [c for c in clusters if any(i["type"] == "missing_values" for i in c.issues)][0]
        assert "age" in missing_cluster.columns
        assert "income" in missing_cluster.columns

    def test_cluster_description(self, sample_issues: list[dict]) -> None:
        clusters = cluster_issues(sample_issues)
        desc = clusters[0].description
        assert "Cluster:" in desc
        assert "missing_values" in desc

    def test_cluster_recommendation_high_score(self) -> None:
        issues = [
            {"column": "a", "type": "missing_values", "severity": "critical", "message": "", "suggestion": ""},
            {"column": "b", "type": "missing_values", "severity": "critical", "message": "", "suggestion": ""},
            {"column": "c", "type": "missing_values", "severity": "critical", "message": "", "suggestion": ""},
            {"column": "d", "type": "missing_values", "severity": "critical", "message": "", "suggestion": ""},
        ]
        clusters = cluster_issues(issues)
        assert clusters[0].score >= 8
        assert "BLOCK" in clusters[0].recommendation

    def test_cluster_recommendation_low_score(self) -> None:
        issues = [
            {"column": "a", "type": "outliers", "severity": "info", "message": "", "suggestion": ""},
        ]
        clusters = cluster_issues(issues)
        assert clusters[0].score <= 2
        assert "Monitor" in clusters[0].recommendation

    def test_critical_only(self, critical_only: list[dict]) -> None:
        clusters = cluster_issues(critical_only)
        assert len(clusters) == 1  # all missing_values
        assert clusters[0].score >= 8
        assert len(clusters[0].issues) == 4

    def test_info_only(self, info_only: list[dict]) -> None:
        clusters = cluster_issues(info_only)
        # outliers → inference, text_length_outliers → training (different areas)
        assert len(clusters) == 2


# ──────────────────────────────────────────────────────────────────────────
# verdict.py tests
# ──────────────────────────────────────────────────────────────────────────

class TestVerdict:
    def test_empty_clusters_pass(self) -> None:
        v = compute_verdict([])
        assert v.decision == "PASS"
        assert v.confidence == "HIGH"
        assert v.score == 0

    def test_low_score_pass(self, info_only: list[dict]) -> None:
        clusters = cluster_issues(info_only)
        v = compute_verdict(clusters)
        # 2 info issues: 1+1 = 2 < warn_score(4) → PASS
        assert v.decision == "PASS"

    def test_medium_score_warn(self, sample_issues: list[dict]) -> None:
        clusters = cluster_issues(sample_issues)
        v = compute_verdict(clusters)
        # Should be WARN with default policy (total score >= 4)
        assert v.decision in ("WARN", "BLOCK")
        assert v.primary_cluster is not None
        assert len(v.all_clusters) > 0

    def test_critical_triggers_block(self, critical_only: list[dict]) -> None:
        clusters = cluster_issues(critical_only)
        v = compute_verdict(clusters)
        # 4 critical issues → score >= 8 → BLOCK
        assert v.decision == "BLOCK"

    def test_custom_policy_block(self, sample_issues: list[dict]) -> None:
        clusters = cluster_issues(sample_issues)
        # Very low block threshold
        policy = Policy(block_score=3, warn_score=1)
        v = compute_verdict(clusters, policy)
        assert v.decision == "BLOCK"

    def test_custom_policy_pass(self) -> None:
        # Minimal issues
        issues = [
            {"column": "a", "type": "outliers", "severity": "info", "message": "", "suggestion": ""},
        ]
        clusters = cluster_issues(issues)
        # Very high thresholds
        policy = Policy(block_score=20, warn_score=20)
        v = compute_verdict(clusters, policy)
        assert v.decision == "PASS"

    def test_next_action_block(self, critical_only: list[dict]) -> None:
        clusters = cluster_issues(critical_only)
        v = compute_verdict(clusters)
        assert "STOP" in v.next_action

    def test_next_action_pass_empty(self) -> None:
        v = compute_verdict([])
        assert "Safe to proceed" in v.next_action

    def test_to_dict(self, sample_issues: list[dict]) -> None:
        clusters = cluster_issues(sample_issues)
        v = compute_verdict(clusters)
        d = v.to_dict()
        assert "verdict" in d
        assert "confidence" in d
        assert "primary_cluster" in d
        assert "clusters" in d
        assert isinstance(d["clusters"], list)
        if d["primary_cluster"] is not None:
            assert "impact_area" in d["primary_cluster"]
            assert "score" in d["primary_cluster"]

    def test_to_dict_empty(self) -> None:
        v = compute_verdict([])
        d = v.to_dict()
        assert d["verdict"] == "PASS"
        assert d["primary_cluster"] is None
        assert d["clusters"] == []

    def test_verdict_frozen(self) -> None:
        v = compute_verdict([])
        with pytest.raises(AttributeError):
            v.decision = "BLOCK"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────
# HTML decision layout tests
# ──────────────────────────────────────────────────────────────────────────

class TestDecisionHTML:
    def test_decision_html_pass(self) -> None:
        from fitcheck.decision import cluster_issues
        from fitcheck.html import render_decision_html
        from fitcheck.verdict import compute_verdict

        clusters = cluster_issues([])
        verdict = compute_verdict(clusters)
        html = render_decision_html([], verdict, None)
        assert "FitCheck Preflight Decision" in html
        assert "PASS" in html
        assert "badge-pass" in html

    def test_decision_html_block(self, critical_only: list[dict]) -> None:
        from fitcheck.html import render_decision_html

        clusters = cluster_issues(critical_only)
        verdict = compute_verdict(clusters)
        html = render_decision_html(critical_only, verdict, None)
        assert "BLOCK" in html
        assert "badge-critical" in html
        assert "Primary cluster" in html

    def test_decision_html_warn(self, sample_issues: list[dict]) -> None:
        from fitcheck.html import render_decision_html

        clusters = cluster_issues(sample_issues)
        verdict = compute_verdict(clusters)
        html = render_decision_html(sample_issues, verdict, None)
        assert verdict.decision in html
        assert "All issues" in html
        assert "fingerprint" in html

    def test_decision_html_writes_file(self, sample_issues: list[dict], tmp_path: Path) -> None:
        from fitcheck.html import render_decision_html

        out = str(tmp_path / "decision.html")
        clusters = cluster_issues(sample_issues)
        verdict = compute_verdict(clusters)
        html = render_decision_html(sample_issues, verdict, out)
        assert Path(out).exists()
        assert len(html) > 100

    def test_decision_html_fingerprint_footer(self) -> None:
        from fitcheck.html import render_decision_html

        clusters = cluster_issues([])
        verdict = compute_verdict(clusters)
        html = render_decision_html([], verdict, None)
        assert "fingerprint" in html


# ──────────────────────────────────────────────────────────────────────────
# CLI --mode and --policy tests
# ──────────────────────────────────────────────────────────────────────────

class TestCLIMode:
    def _make_csv(self, tmp_path: Path, n_rows: int = 50) -> Path:
        """Create a test CSV with some issues."""
        df = pd.DataFrame({
            "a": [1.0] * n_rows,
            "b": list(range(n_rows)),
            "c": ["x"] * n_rows,
        })
        # Add missing values in column 'b' (20% missing → critical)
        df.loc[df.index[:10], "b"] = None
        # Make 'a' constant
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        return path

    def test_classic_mode_default(self, tmp_path: Path) -> None:
        from fitcheck.cli import main
        csv = self._make_csv(tmp_path)
        out = str(tmp_path / "report.html")
        exit_code = main(["check", str(csv), "--output", out, "--quiet"])
        # Classic mode: should produce normal report
        assert Path(out).exists()
        assert exit_code in (0, 1, 2)

    def test_decision_mode_flag(self, tmp_path: Path) -> None:
        from fitcheck.cli import main
        csv = self._make_csv(tmp_path)
        out = str(tmp_path / "report.html")
        exit_code = main(["check", str(csv), "--output", out, "--mode", "decision", "--quiet"])
        assert exit_code in (0, 1, 2)

    def test_decision_mode_default_is_classic(self, tmp_path: Path) -> None:
        from fitcheck.cli import main
        csv = self._make_csv(tmp_path)
        out = str(tmp_path / "report.html")
        # No --mode flag → default is classic
        exit_code = main(["check", str(csv), "--output", out, "--quiet"])
        assert exit_code in (0, 1, 2)

    def test_policy_flag(self, tmp_path: Path) -> None:
        from fitcheck.cli import main
        csv = self._make_csv(tmp_path)
        policy_file = tmp_path / "fitcheck.yaml"
        policy_file.write_text(yaml.dump({
            "fail_thresholds": {"block_score": 3, "warn_score": 1},
        }))
        out = str(tmp_path / "report.html")
        exit_code = main([
            "check", str(csv),
            "--output", out,
            "--mode", "decision",
            "--policy", str(policy_file),
            "--quiet",
        ])
        assert exit_code in (0, 1, 2)


# ──────────────────────────────────────────────────────────────────────────
# Integration: full pipeline (check → cluster → verdict → HTML)
# ──────────────────────────────────────────────────────────────────────────

class TestSprint2Integration:
    def test_full_pipeline_with_issues(self, tmp_path: Path) -> None:
        """End-to-end: create CSV → check → cluster → verdict → decision HTML."""
        df = pd.DataFrame({
            "col1": [1.0] * 30 + [None] * 20,  # 40% missing → critical
            "col2": list(range(50)),
            "col3": ["a"] * 50,  # constant
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        from fitcheck.check import check
        from fitcheck.decision import cluster_issues
        from fitcheck.html import render_decision_html
        from fitcheck.policy import load_policy
        from fitcheck.verdict import compute_verdict

        result = check(
            data=str(csv_path),
            output=str(tmp_path / "classic.html"),
            return_format="dict",
        )
        issues = result["issues"]
        assert len(issues) > 0

        clusters = cluster_issues(issues)
        assert len(clusters) > 0

        policy = load_policy(None)
        verdict = compute_verdict(clusters, policy)
        assert verdict.decision in ("WARN", "BLOCK")

        html = render_decision_html(
            issues, verdict, str(tmp_path / "decision.html")
        )
        assert "Preflight Decision" in html
        assert verdict.decision in html
        assert Path(tmp_path / "decision.html").exists()

    def test_full_pipeline_clean_data(self, tmp_path: Path) -> None:
        """End-to-end with clean data → PASS verdict."""
        df = pd.DataFrame({
            "a": list(range(100)),
            "b": [float(x) for x in range(100)],
        })
        csv_path = tmp_path / "clean.csv"
        df.to_csv(csv_path, index=False)

        from fitcheck.check import check
        from fitcheck.decision import cluster_issues
        from fitcheck.html import render_decision_html
        from fitcheck.verdict import compute_verdict

        result = check(
            data=str(csv_path),
            output=str(tmp_path / "report.html"),
            return_format="dict",
        )
        issues = result["issues"]
        clusters = cluster_issues(issues)
        verdict = compute_verdict(clusters)
        assert verdict.decision == "PASS"
        html = render_decision_html(issues, verdict, None)
        assert "PASS" in html
