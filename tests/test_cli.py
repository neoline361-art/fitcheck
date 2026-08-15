"""Tests for the fitcheck command-line interface."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fitcheck.cli import main


class TestCli:
    """Tests for CLI command paths."""

    def test_no_command_prints_help(self, capsys) -> None:
        """Running with no arguments prints help and returns 1."""
        result = main([])
        captured = capsys.readouterr()
        assert result == 1
        assert "usage" in captured.out.lower()

    def test_check_command_generates_report(self, tmp_path: Path) -> None:
        """`fitcheck check` runs against a CSV and writes an HTML report."""
        csv = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]}).to_csv(csv, index=False)
        out = str(tmp_path / "report.html")
        result = main(["check", str(csv), "--output", out])
        assert result == 2  # 33% missing in "a" exceeds the 20% critical threshold
        assert Path(out).exists()
        assert "FitCheck" in Path(out).read_text()

    def test_check_command_with_target_and_auto_fix(self, tmp_path: Path) -> None:
        """`fitcheck check` honors --target and --auto-fix."""
        csv = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "feat": [1, 2, 3, 4, 5],
                "label": [0, 0, 0, 0, 1],  # 80% imbalance
            }
        ).to_csv(csv, index=False)
        out = str(tmp_path / "report.html")
        result = main(["check", str(csv), "--target", "label", "--output", out, "--auto-fix"])
        assert result == 1  # 80% class imbalance is a warning under CI exit codes
        assert Path(out).exists()
        # Fix script is generated next to the report
        assert (tmp_path / "report_fix_script.py").exists()

    def test_check_command_missing_file(self) -> None:
        """A nonexistent path is a runtime error and exits with code 3."""
        assert main(["check", "/nonexistent/path.csv"]) == 3

    def test_check_json_output(self, tmp_path: Path, capsys) -> None:
        """`--json` emits machine-readable results to stdout."""
        import json

        csv = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(csv, index=False)
        out = str(tmp_path / "report.html")
        assert main(["check", str(csv), "--output", out, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "issues" in payload
        assert "total_rows" in payload
        assert payload["total_rows"] == 3

    def test_check_fail_on_exit_codes(self, tmp_path: Path) -> None:
        """`--fail-on` gates the exit code by severity."""
        csv = tmp_path / "data.csv"
        pd.DataFrame(
            {"feat": [1, 2, 3, 4, 5], "label": [0, 0, 0, 0, 1]}  # 80% imbalance -> warning
        ).to_csv(csv, index=False)
        out = str(tmp_path / "report.html")
        assert main(["check", str(csv), "--target", "label", "--output", out]) == 1  # warnings fail by default
        assert main(["check", str(csv), "--target", "label", "--output", out, "--fail-on", "critical"]) == 0

        critical = tmp_path / "critical.csv"
        pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]}).to_csv(critical, index=False)
        assert main(["check", str(critical), "--output", out, "--fail-on", "critical"]) == 2

    def test_check_multi_file(self, tmp_path: Path, monkeypatch) -> None:
        """Multiple data files produce one report each."""
        monkeypatch.chdir(tmp_path)
        first = tmp_path / "a.csv"
        second = tmp_path / "b.csv"
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(first, index=False)
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(second, index=False)
        assert main(["check", str(first), str(second)]) == 0
        assert (tmp_path / "fitcheck_report_a.html").exists()
        assert (tmp_path / "fitcheck_report_b.html").exists()

    def test_check_quiet_suppresses_stdout(self, tmp_path: Path, capsys) -> None:
        """`--quiet` suppresses everything except the exit code."""
        csv = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)
        assert main(["check", str(csv), "--output", str(tmp_path / "r.html"), "--quiet"]) == 0
        assert capsys.readouterr().out == ""

    def test_check_time_column_flag(self, tmp_path: Path) -> None:
        """`--time-column` runs time-series checks without failing clean data."""
        csv = tmp_path / "ts.csv"
        pd.DataFrame(
            {"ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), "x": [1, 2, 3]}
        ).to_csv(csv, index=False)
        out = str(tmp_path / "report.html")
        assert main(["check", str(csv), "--time-column", "ts", "--output", out, "--fail-on", "critical"]) == 0

    def test_check_plugins_flag(self, tmp_path: Path, monkeypatch) -> None:
        """`--plugins` loads a dotted-module check and its issues affect the exit code."""
        (tmp_path / "cli_checkmod.py").write_text(
            "def check(df):\n    return [{'column': 'x', 'type': 'custom', 'severity': 'critical', 'message': 'blocked', 'suggestion': 'fix'}]\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        csv = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)
        out = str(tmp_path / "report.html")
        assert main(["check", str(csv), "--plugins", "cli_checkmod", "--output", out, "--quiet"]) == 2

    def test_report_command_generates_report(self, tmp_path: Path) -> None:
        """`fitcheck report` loads a pickled model + arrays and writes HTML."""
        np.random.seed(42)
        x = np.random.randn(30, 3)
        y = (x[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(x, y)

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        x_path = tmp_path / "X_test.npy"
        y_path = tmp_path / "y_test.npy"
        np.save(x_path, x)
        np.save(y_path, y)

        out = str(tmp_path / "model.html")
        result = main(
            [
                "report",
                str(model_path),
                str(x_path),
                str(y_path),
                "--output",
                out,
            ]
        )
        assert result == 0
        assert Path(out).exists()
        assert "FitCheck Model Report" in Path(out).read_text()

    def test_drift_command_generates_report(self, tmp_path: Path) -> None:
        """`fitcheck drift` compares two CSVs and writes an HTML report."""
        ref = tmp_path / "ref.csv"
        prod = tmp_path / "prod.csv"
        pd.DataFrame({"feat": np.random.randn(200)}).to_csv(ref, index=False)
        pd.DataFrame({"feat": np.random.randn(200) + 5}).to_csv(prod, index=False)
        out = str(tmp_path / "drift.html")
        result = main(["drift", str(ref), str(prod), "--output", out])
        assert result == 0
        assert Path(out).exists()
        assert "FitCheck Drift Report" in Path(out).read_text()
