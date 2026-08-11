from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fitcheck.check import check
from fitcheck.drift import detect_drift
from fitcheck.report import report


def test_check_accepts_threshold_overrides(tmp_path: Path) -> None:
    data = pd.DataFrame({"x": [1, None, 3, 4], "y": [0, 1, 0, 1]})
    result = check(data, output=str(tmp_path / "check.html"), return_format="dict", config={"missing_warning": 0.50, "missing_critical": 0.75})
    assert result["config"]["missing_warning"] == 0.50
    assert result["passed"] is True


def test_drift_supports_psi_and_wasserstein(tmp_path: Path) -> None:
    ref = pd.DataFrame({"x": np.arange(100, dtype=float)})
    prod = pd.DataFrame({"x": np.arange(100, dtype=float) + 20})
    psi = detect_drift(ref, prod, output=str(tmp_path / "psi.html"), method="psi")
    wasserstein = detect_drift(ref, prod, output=str(tmp_path / "wasserstein.html"), method="wasserstein")
    assert psi[0]["test"] == "PSI"
    assert wasserstein[0]["test"] == "Wasserstein"
    assert psi[0]["drifted"] and wasserstein[0]["drifted"]


def test_model_report_adds_pr_metrics_without_dataframe_warning(tmp_path: Path) -> None:
    x = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1], "b": [1, 1, 0, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = RandomForestClassifier(n_estimators=5, random_state=1).fit(x, y)
    result = report(model, x, y, output=str(tmp_path / "model.html"))
    assert "average_precision" in result
    assert "recommended_threshold" in result
    assert (tmp_path / "model.html").exists()


def test_full_cli_workflow(tmp_path: Path) -> None:
    from fitcheck.cli import main

    x = pd.DataFrame({"a": [0, 1, 0, 1], "target": [0, 1, 0, 1]})
    data_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.pkl"
    x.to_csv(data_path, index=False)
    model = RandomForestClassifier(n_estimators=3, random_state=1).fit(x[["a"]], x["target"])
    with model_path.open("wb") as file:
        pickle.dump(model, file)
    assert main(["full", str(data_path), "--target", "target", "--model", str(model_path), "--output-dir", str(tmp_path / "reports")]) == 0
    assert (tmp_path / "reports" / "dataset_report.html").exists()
    assert (tmp_path / "reports" / "model_report.html").exists()


def test_timeseries_and_plugin_extensions(tmp_path: Path) -> None:
    data = pd.DataFrame({"timestamp": ["2024-01-02", "bad", "2024-01-01"], "x": [1, 2, 3]})

    def custom_check(frame: pd.DataFrame) -> list[dict[str, object]]:
        return [{"type": "custom", "severity": "info", "message": "plugin ran"}]

    result = check(
        data,
        output=str(tmp_path / "extensions.html"),
        return_format="dict",
        plugins=[custom_check],
        time_column="timestamp",
    )
    assert any(issue["type"] == "custom" for issue in result["issues"])
    assert any(issue["type"] == "invalid_timestamps" for issue in result["issues"])
    assert any(issue["type"] == "non_monotonic_time" for issue in result["issues"])


def test_check_can_sample_large_csv(tmp_path: Path) -> None:
    path = tmp_path / "contacts.csv"
    pd.DataFrame({"mobile": ["+12025550101"] * 20, "name": ["Person"] * 20}).to_csv(path, index=False)
    result = check(path.as_posix(), output=str(tmp_path / "sample.html"), return_format="dict", sample_rows=5)
    assert result["sampled"] is True
    assert result["sample_rows"] == 5
    assert result["total_rows"] == 5
