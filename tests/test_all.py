"""Comprehensive test suite for FitCheck v2.0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

import fitcheck
from fitcheck.check import check
from fitcheck.drift import detect_drift
from fitcheck.fix import FixAction, FixScriptGenerator, generate_fix_script
from fitcheck.report import report

# ---------------------------------------------------------------------------
# check.py tests
# ---------------------------------------------------------------------------


class TestCheck:
    """Tests for the dataset health check engine."""

    def test_check_valid_csv(self, tmp_path: Path) -> None:
        """Normal CSV loads and generates a report."""
        csv = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(csv, index=False)
        result = check(str(csv), output=str(tmp_path / "report.html"))
        assert isinstance(result, list)
        assert len(result) == 0  # Clean data

    def test_check_missing_file(self) -> None:
        """FileNotFoundError raised for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            check("/nonexistent/path.csv")

    def test_check_empty_dataframe(self, tmp_path: Path) -> None:
        """Empty DataFrame (headers only) handles gracefully."""
        csv = tmp_path / "empty.csv"
        pd.DataFrame(columns=["a", "b"]).to_csv(csv, index=False)
        result = check(str(csv), output=str(tmp_path / "report.html"))
        assert isinstance(result, list)

    def test_check_single_row(self, tmp_path: Path) -> None:
        """Single row dataset handles without division by zero."""
        csv = tmp_path / "single.csv"
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv, index=False)
        result = check(str(csv), output=str(tmp_path / "report.html"))
        assert isinstance(result, list)

    def test_check_all_null_column(self, tmp_path: Path) -> None:
        """All-null column flags critical severity."""
        csv = tmp_path / "nulls.csv"
        pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]}).to_csv(csv, index=False)
        result = check(str(csv), output=str(tmp_path / "report.html"))
        assert any(i.get("severity") == "critical" for i in result)

    def test_check_no_issues(self, tmp_path: Path) -> None:
        """Clean data returns PASS status."""
        csv = tmp_path / "clean.csv"
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})
        df.to_csv(csv, index=False)
        result = check(str(csv), output=str(tmp_path / "report.html"), return_format="dict")
        assert isinstance(result, dict)
        assert result["passed"] is True
        assert result["summary"]["critical"] == 0

    def test_check_dict_output(self, tmp_path: Path) -> None:
        """Dict output has correct keys."""
        csv = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2, None]}).to_csv(csv, index=False)
        result = check(str(csv), output=str(tmp_path / "report.html"), return_format="dict")
        assert all(
            k in result for k in ("total_rows", "total_columns", "issues", "passed", "summary")
        )

    def test_check_auto_fix_generates_script(self, tmp_path: Path) -> None:
        """auto_fix=True creates a fix script file."""
        csv = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2, None]}).to_csv(csv, index=False)
        out = str(tmp_path / "report.html")
        check(str(csv), output=out, auto_fix=True)
        script = tmp_path / "report_fix_script.py"
        assert script.exists() or (tmp_path / "fitcheck_fix_script.py").exists()

    def test_check_with_target(self, tmp_path: Path) -> None:
        """Target column triggers imbalance + outlier checks."""
        csv = tmp_path / "data.csv"
        df = pd.DataFrame(
            {
                "feat": [1, 2, 3, 4, 5],
                "label": [0, 0, 0, 0, 1],  # 80% imbalance
            }
        )
        df.to_csv(csv, index=False)
        result = check(str(csv), target="label", output=str(tmp_path / "report.html"))
        assert any(i.get("type") == "class_imbalance" for i in result)

    def test_check_duplicate_rows(self, tmp_path: Path) -> None:
        """Duplicate rows are detected."""
        csv = tmp_path / "dups.csv"
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        df.to_csv(csv, index=False)
        result = check(str(csv), output=str(tmp_path / "report.html"))
        assert any(i.get("type") == "duplicate_rows" for i in result)

    def test_check_outlier_denominator_uses_non_null(self, tmp_path: Path) -> None:
        """Outlier ratio is computed against non-null values, not total rows."""
        # 1 outlier among 10 non-null values (10%) — must be flagged even though
        # it is only 5% of total rows (1/20) due to NaNs.
        csv = tmp_path / "outliers.csv"
        df = pd.DataFrame(
            {
                "a": [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    1000.0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
            }
        )
        df.to_csv(csv, index=False)
        result = check(str(csv), output=str(tmp_path / "report.html"))
        outlier_issue = [i for i in result if i.get("type") == "outliers"]
        assert outlier_issue, "outlier should be flagged based on non-null denominator"
        assert "10.0%" in outlier_issue[0]["message"]


# ---------------------------------------------------------------------------
# report.py tests
# ---------------------------------------------------------------------------


class TestReport:
    """Tests for the model evaluation engine."""

    def test_classification_report(self, tmp_path: Path) -> None:
        """RandomForest classification produces correct metrics."""
        np.random.seed(42)
        x = np.random.randn(100, 5)
        y = (x[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(x, y)
        metrics = report(model, x, y, output=str(tmp_path / "report.html"))
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] > 0

    def test_regression_report(self, tmp_path: Path) -> None:
        """LinearRegression produces correct metrics."""
        np.random.seed(42)
        x = np.random.randn(50, 3)
        y = 3 * x[:, 0] + 2 * x[:, 1] + np.random.randn(50) * 0.1
        model = LinearRegression()
        model.fit(x, y)
        metrics = report(model, x, y, output=str(tmp_path / "report.html"))
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_report_html_created(self, tmp_path: Path) -> None:
        """HTML report file is created."""
        np.random.seed(42)
        x = np.random.randn(20, 3)
        y = (x[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(x, y)
        out = str(tmp_path / "model.html")
        report(model, x, y, output=out)
        assert Path(out).exists()
        content = Path(out).read_text()
        assert "FitCheck Model Report" in content

    def test_report_feature_importance(self, tmp_path: Path) -> None:
        """Tree models include feature importance."""
        np.random.seed(42)
        x = np.random.randn(30, 5)
        y = (x[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(x, y)
        metrics = report(model, x, y, output=str(tmp_path / "report.html"))
        assert "feature_importance" in metrics

    def test_report_feature_importance_uses_column_names(self, tmp_path: Path) -> None:
        """Feature importance uses real DataFrame column names."""
        np.random.seed(42)
        x = pd.DataFrame(np.random.randn(30, 3), columns=["age", "income", "score"])
        y = (x["age"] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(x, y)
        metrics = report(model, x, y, output=str(tmp_path / "report.html"))
        importance = metrics["feature_importance"]
        assert set(importance.keys()) == {"age", "income", "score"}
        assert all("feature_" not in k for k in importance)


# ---------------------------------------------------------------------------
# drift.py tests
# ---------------------------------------------------------------------------


class TestDrift:
    """Tests for the drift detection engine."""

    def test_no_drift(self, tmp_path: Path) -> None:
        """Identical distributions produce zero drift."""
        np.random.seed(42)
        ref = pd.DataFrame({"a": np.random.randn(200)})
        prod = pd.DataFrame({"a": np.random.randn(200)})
        results = detect_drift(ref, prod, output=str(tmp_path / "drift.html"))
        assert sum(r["drifted"] for r in results) == 0

    def test_drift_detected(self, tmp_path: Path) -> None:
        """Shifted mean produces drift detection."""
        np.random.seed(42)
        ref = pd.DataFrame({"a": np.random.randn(200)})
        prod = pd.DataFrame({"a": np.random.randn(200) + 5})
        results = detect_drift(ref, prod, output=str(tmp_path / "drift.html"))
        assert sum(r["drifted"] for r in results) >= 1

    def test_categorical_drift(self, tmp_path: Path) -> None:
        """Different category distributions produce drift."""
        ref = pd.DataFrame({"cat": ["a"] * 80 + ["b"] * 20})
        prod = pd.DataFrame({"cat": ["a"] * 20 + ["b"] * 80})
        results = detect_drift(ref, prod, output=str(tmp_path / "drift.html"))
        assert sum(r["drifted"] for r in results) >= 1

    def test_drift_html_created(self, tmp_path: Path) -> None:
        """Drift HTML report is generated."""
        ref = pd.DataFrame({"a": [1, 2, 3]})
        prod = pd.DataFrame({"a": [1, 2, 3]})
        out = str(tmp_path / "drift.html")
        detect_drift(ref, prod, output=out)
        assert Path(out).exists()
        assert "FitCheck Drift Report" in Path(out).read_text()


# ---------------------------------------------------------------------------
# fix.py tests — the killer feature
# ---------------------------------------------------------------------------


class TestAutoFix:
    """Tests for the transparent fix script generator."""

    def test_script_contains_header(self) -> None:
        """Generated script contains warning header."""
        diagnostics = {
            "issues": [
                {"column": "a", "type": "missing_values", "severity": "warning", "message": "test"}
            ]
        }
        script = generate_fix_script(diagnostics, "input.csv", "/tmp/test_script.py")
        assert "Review every step before running" in script
        assert "WARNING" in script

    def test_never_overwrites_input(self) -> None:
        """Script uses different paths for input and output."""
        diagnostics = {"issues": []}
        script = generate_fix_script(diagnostics, "data.csv", "/tmp/test_script.py")
        assert "INPUT_PATH" in script
        assert "OUTPUT_PATH" in script
        # Input and output should differ
        lines = script.split("\n")
        input_line = [line for line in lines if "INPUT_PATH =" in line][0]
        output_line = [line for line in lines if "OUTPUT_PATH =" in line][0]
        assert input_line != output_line

    def test_idempotent_output(self) -> None:
        """Same inputs produce the same script (except timestamp)."""
        diagnostics = {
            "issues": [
                {"column": "x", "type": "constant_column", "severity": "warning", "message": "test"}
            ]
        }
        script1 = generate_fix_script(diagnostics, "in.csv", "/tmp/t1.py")
        script2 = generate_fix_script(diagnostics, "in.csv", "/tmp/t2.py")
        # Strip timestamps for comparison
        s1 = "\n".join(line for line in script1.split("\n") if "Generated:" not in line)
        s2 = "\n".join(line for line in script2.split("\n") if "Generated:" not in line)
        assert s1 == s2

    def test_missing_values_action(self) -> None:
        """Missing values issue generates median imputation code."""
        diagnostics = {
            "issues": [
                {
                    "column": "age",
                    "type": "missing_values",
                    "severity": "warning",
                    "message": "50% missing",
                }
            ]
        }
        script = generate_fix_script(diagnostics, "data.csv", "/tmp/test.py")
        assert "fillna" in script
        assert "median" in script

    def test_duplicate_rows_action(self) -> None:
        """Duplicate rows issue generates drop_duplicates code."""
        diagnostics = {
            "issues": [
                {
                    "column": "all",
                    "type": "duplicate_rows",
                    "severity": "warning",
                    "message": "5 dups",
                }
            ]
        }
        script = generate_fix_script(diagnostics, "data.csv", "/tmp/test.py")
        assert "drop_duplicates" in script

    def test_constant_column_action(self) -> None:
        """Constant column issue generates drop code."""
        diagnostics = {
            "issues": [
                {
                    "column": "useless",
                    "type": "constant_column",
                    "severity": "warning",
                    "message": "all same",
                }
            ]
        }
        script = generate_fix_script(diagnostics, "data.csv", "/tmp/test.py")
        assert "drop" in script
        assert "useless" in script

    def test_fix_action_dataclass(self) -> None:
        """FixAction is frozen and hashable."""
        action = FixAction(
            column="test",
            issue_type="missing_values",
            severity="warning",
            description="test desc",
            code="pass",
            rationale="test rationale",
        )
        assert action.column == "test"
        # Frozen dataclass should not be modifiable
        with pytest.raises(AttributeError):
            action.column = "other"

    def test_generator_save(self, tmp_path: Path) -> None:
        """FixScriptGenerator.save writes a file."""
        gen = FixScriptGenerator()
        gen.add(
            FixAction(
                column="a",
                issue_type="test",
                severity="info",
                description="d",
                code="pass",
                rationale="r",
            )
        )
        path = gen.save("in.csv", str(tmp_path / "fix.py"), "out.csv")
        assert path.exists()


# ---------------------------------------------------------------------------
# Integration / sanity tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_version(self) -> None:
        """Version is accessible and single-sourced."""
        from fitcheck._version import __version__ as source_version

        assert fitcheck.__version__ == "3.1.3"
        assert fitcheck.__version__ == source_version
        assert "check" in fitcheck.__all__
        assert "report" in fitcheck.__all__
        assert "detect_drift" in fitcheck.__all__
        assert "registry" in fitcheck.__all__
        assert "load_plugin" in fitcheck.__all__

    def test_package_imports(self) -> None:
        """All public functions are importable."""
        from fitcheck import check, detect_drift, report

        assert callable(check)
        assert callable(report)
        assert callable(detect_drift)

    def test_cli_help(self) -> None:
        """CLI --help prints usage without errors."""
        from fitcheck.cli import main

        try:
            main(["--help"])
        except SystemExit as e:
            assert e.code == 0

    def test_cli_demo(self) -> None:
        """CLI demo runs via main() without crashing (headless)."""
        from fitcheck.cli import main

        result = main(["demo", "--no-browser"])
        assert result == 0

    def test_main_module_runs(self, monkeypatch, capsys) -> None:
        """`python -m fitcheck` prints help and exits cleanly."""
        import runpy

        monkeypatch.setattr("sys.argv", ["fitcheck"])
        runpy.run_module("fitcheck.__main__", run_name="__main__")
        assert "usage" in capsys.readouterr().out.lower()

    def test_pro_module_importable(self) -> None:
        """The pro shim exposes the fix script API."""
        from fitcheck.pro import FixScriptGenerator, generate_fix_script

        assert callable(FixScriptGenerator)
        assert callable(generate_fix_script)
