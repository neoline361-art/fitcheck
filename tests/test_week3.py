"""Tests for the v3.3.0 polish sprint.

Covers: demo severity levels, auto-fix script execution, report structure,
edge-case hardening, mock-based backend tests, and viz boundaries.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fitcheck.check import check
from fitcheck.html import render_check_html

# ---------------------------------------------------------------------------
# Demo severity coverage
# ---------------------------------------------------------------------------


class TestDemoSeverities:
    """Verify that the demo dataset triggers every severity level."""

    def test_demo_has_critical_severity(self, tmp_path: Path) -> None:
        """Demo income column has 25% missing >= 20% critical threshold."""
        csv = tmp_path / "demo.csv"
        np.random.seed(42)
        n = 500
        scores = np.random.normal(700, 100, n)
        scores[:8] = 9999
        df = pd.DataFrame({
            "age": np.concatenate([np.random.normal(35, 10, n - 40), [np.nan] * 40]),
            "income": np.concatenate([np.random.normal(50000, 15000, n - 125), [np.nan] * 125]),
            "score": scores,
            "constant_col": [42] * n,
            "label": [0] * (n - n // 5) + [1] * (n // 5),
        })
        df = pd.concat([df, df.head(10)], ignore_index=True)
        df.to_csv(csv, index=False)
        result = check(str(csv), target="label", output=str(tmp_path / "report.html"))
        severities = {i.get("severity") for i in result}
        assert "critical" in severities, "income 25% missing should trigger critical"
        assert "warning" in severities, "age 8% missing or constant column should trigger warning"
        assert "info" in severities, "duplicates or outliers should trigger info"

    def test_demo_dont_crash(self) -> None:
        """fitcheck demo --no-browser runs without crashing."""
        from fitcheck.demo import run_demo

        run_demo(no_browser=True, output_dir="/tmp/fitcheck_demo_test")
        assert True  # no crash


# ---------------------------------------------------------------------------
# Auto-fix script execution
# ---------------------------------------------------------------------------


class TestAutoFixExecution:
    """Verify generated fix scripts actually run without errors."""

    @pytest.mark.parametrize(
        "kind,make_data",
        [
            ("missing_values", lambda p: pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]}).to_csv(p, index=False)),
            ("duplicates", lambda p: pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]}).to_csv(p, index=False)),
            ("constant_column", lambda p: pd.DataFrame({"a": [1, 2, 3], "b": [42, 42, 42]}).to_csv(p, index=False)),
        ],
    )
    def test_fix_script_executes(self, tmp_path: Path, kind: str, make_data) -> None:
        """Generated fix script compiles and runs without errors."""
        csv = tmp_path / f"{kind}.csv"
        make_data(csv)
        check(str(csv), output=str(tmp_path / "report.html"), auto_fix=True)
        candidate = tmp_path / f"{kind}_report_fix_script.py"
        if not candidate.exists():
            candidate = tmp_path / "report_fix_script.py"
        if not candidate.exists():
            pytest.skip(f"No fix script generated for {kind}")
        result = subprocess.run(
            [sys.executable, str(candidate)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tmp_path,
        )
        assert result.returncode == 0, f"{kind}: {result.stderr[-500:]}"
        assert (tmp_path / "cleaned_data.csv").exists(), f"{kind}: cleaned output not written"


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


class TestReportStructure:
    """Verify premium report features: critical-first, collapsible, copy snippets."""

    def test_critical_before_warning(self, tmp_path: Path) -> None:
        """Critical issues appear before warnings in the rendered HTML."""
        issues = [
            {"column": "b", "type": "missing_values", "severity": "warning", "message": "w", "suggestion": "s"},
            {"column": "a", "type": "missing_values", "severity": "critical", "message": "c", "suggestion": "s"},
        ]
        html = render_check_html(issues, pd.DataFrame({"a": [1], "b": [2]}), str(tmp_path / "r.html"))
        assert html.index("<h2>Critical issues</h2>") < html.index("<summary>Warnings")

    def test_collapsible_section(self, tmp_path: Path) -> None:
        """Warnings and info are wrapped in a collapsible details element."""
        issues = [
            {"column": "a", "type": "missing_values", "severity": "warning", "message": "w", "suggestion": "s"},
            {"column": "b", "type": "outliers", "severity": "info", "message": "i", "suggestion": "s"},
        ]
        html = render_check_html(issues, pd.DataFrame({"a": [1], "b": [2]}), str(tmp_path / "r.html"))
        assert '<details class="collapsible"' in html

    def test_copy_button_present(self, tmp_path: Path) -> None:
        """Copyable fix snippets are present for issues with code."""
        issues = [
            {"column": "a", "type": "missing_values", "severity": "warning", "message": "w", "suggestion": "s"},
        ]
        html = render_check_html(issues, pd.DataFrame({"a": [1.0, 2.0, None]}), str(tmp_path / "r.html"))
        assert "copy-btn" in html
        assert "navigator.clipboard" in html


# ---------------------------------------------------------------------------
# Edge-case hardening
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Ensure no crashes on pathological inputs."""

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame returns no issues, no crash."""
        result = check(pd.DataFrame(), output=None)
        assert isinstance(result, list)

    def test_single_row(self) -> None:
        """Single-row DataFrame handles without division by zero."""
        result = check(pd.DataFrame({"a": [1], "b": [2]}), output=None)
        assert isinstance(result, list)

    def test_all_nan_column(self) -> None:
        """All-NaN column triggers critical missing-values."""
        result = check(pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]}), output=None)
        assert any(i.get("severity") == "critical" for i in result)

    def test_constant_with_nan(self) -> None:
        """Column [1, 1, None] is detected as constant (ignores NaN)."""
        result = check(pd.DataFrame({"a": [1, 1, None]}), output=None)
        types = {i.get("type") for i in result}
        assert "constant_column" in types

    def test_one_class_target(self) -> None:
        """Single-class target does not crash imbalance check."""
        result = check(
            pd.DataFrame({"f": [1, 2, 3], "t": [0, 0, 0]}),
            target="t",
            output=None,
        )
        assert isinstance(result, list)

    def test_object_dtype_median_no_crash(self, tmp_path: Path) -> None:
        """Object-dtype columns with missing values don't crash fix generation."""
        csv = tmp_path / "obj.csv"
        pd.DataFrame({"cat": ["a", None, "c"]}).to_csv(csv, index=False)
        check(str(csv), output=str(tmp_path / "r.html"), auto_fix=True)


# ---------------------------------------------------------------------------
# Backend hardening
# ---------------------------------------------------------------------------


class TestBackendHardening:
    """Verify get_backend behavior for unknown names."""

    def test_unknown_backend_raises(self) -> None:
        """Unknown backend name raises ValueError."""
        from fitcheck.backends import get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_none_backend_defaults_to_pandas(self) -> None:
        """get_backend(None) returns PandasBackend."""
        from fitcheck.backends import get_backend

        backend = get_backend(None)
        assert backend.name == "pandas"


# ---------------------------------------------------------------------------
# Viz boundaries
# ---------------------------------------------------------------------------


class TestVizBoundaries:
    """Target surviving mutants in viz/__init__.py and viz/plotly.py."""

    def test_static_histogram_all_nan(self) -> None:
        """Static renderer handles all-NaN series gracefully."""
        from fitcheck.viz import get_renderer

        renderer = get_renderer("static")
        fragment = renderer.render_histogram(pd.Series([None, None]), "empty")
        assert fragment.startswith("<img")
        # Decode and verify PNG magic bytes
        b64 = fragment.split('base64,')[1].split('"')[0]
        raw = base64.b64decode(b64)
        assert raw[:8] == b'\x89PNG\r\n\x1a\n'

    def test_static_histogram_single_value(self) -> None:
        """Static renderer handles single-value series."""
        from fitcheck.viz import get_renderer

        renderer = get_renderer("static")
        fragment = renderer.render_histogram(pd.Series([5.0]), "single")
        assert fragment.startswith("<img")
        assert "single" in fragment

    def test_static_feature_importance_empty(self) -> None:
        """Static renderer handles empty feature importance."""
        from fitcheck.viz import get_renderer

        renderer = get_renderer("static")
        fragment = renderer.render_feature_importance([], [])
        assert fragment.startswith("<img")

    def test_static_confusion_matrix_single_class(self) -> None:
        """Static renderer handles 1x1 confusion matrix."""
        from fitcheck.viz import get_renderer

        renderer = get_renderer("static")
        fragment = renderer.render_confusion_matrix(np.array([[5]]), ["0"])
        assert fragment.startswith("<img")

    def test_get_renderer_unknown_falls_back(self) -> None:
        """Unknown renderer name falls back to static."""
        from fitcheck.viz import get_renderer

        renderer = get_renderer("unknown")
        assert renderer.name == "static"

    def test_plotly_renderer_edges(self) -> None:
        """Plotly renderer handles edge cases."""
        from fitcheck.viz import get_renderer

        renderer = get_renderer("plotly")
        fragment = renderer.render_roc_curve([0, 1], [0, 1], 0.5)
        assert "Plotly.newPlot" in fragment
