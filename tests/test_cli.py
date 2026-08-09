"""Tests for the fitcheck command-line interface."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
        assert result == 0
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
        assert result == 0
        assert Path(out).exists()
        # Fix script is generated next to the report
        assert (tmp_path / "report_fix_script.py").exists()

    def test_check_command_missing_file(self) -> None:
        """A nonexistent path propagates FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            main(["check", "/nonexistent/path.csv"])

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
