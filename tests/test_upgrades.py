from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def test_high_cardinality_detected(tmp_path: Path) -> None:
    data = pd.DataFrame({"id": list(range(50)), "grp": ["A", "B"] * 25})
    result = check(data, output=str(tmp_path / "check.html"), return_format="dict")
    types = [issue["type"] for issue in result["issues"]]
    assert "high_cardinality" in types
    assert "high_cardinality" not in [i["type"] for i in result["issues"] if i["column"] == "grp"]


def test_text_length_skew_detected(tmp_path: Path) -> None:
    data = pd.DataFrame({"text": ["ok"] * 20 + ["x" * 200] * 3})
    result = check(data, output=str(tmp_path / "check.html"), return_format="dict")
    assert any(issue["type"] == "text_length_outliers" for issue in result["issues"])


def test_timeseries_gap_detected(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 03:00", "2024-01-01 04:00", "2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00"]
            ),
            "x": list(range(8)),
        }
    )
    result = check(data, output=str(tmp_path / "check.html"), return_format="dict", time_column="ts")
    assert any(issue["type"] == "time_series_gaps" and issue["severity"] == "critical" for issue in result["issues"])


def test_drift_js_method(tmp_path: Path) -> None:
    ref = pd.DataFrame({"x": np.random.default_rng(1).normal(0, 1, 2000)})
    prod = pd.DataFrame({"x": np.random.default_rng(2).normal(5, 1, 2000)})
    result = detect_drift(ref, prod, output=str(tmp_path / "js.html"), method="js")
    assert result[0]["test"] == "JS"
    assert result[0]["drifted"] is True


def test_drift_schema_drift_detected(tmp_path: Path) -> None:
    ref = pd.DataFrame({"a": [1, 2], "extra": [3, 4]})
    prod = pd.DataFrame({"a": ["1", "2"]})  # dtype change + missing column
    result = detect_drift(ref, prod, output=str(tmp_path / "schema.html"))
    schema_issues = [r for r in result if r.get("type") == "schema"]
    assert any("missing in production" in r["message"] for r in schema_issues)
    assert any("dtype changed" in r["message"] for r in schema_issues)
    assert all(r["severity"] == "critical" for r in schema_issues)


def test_report_binary_calibration_metrics(tmp_path: Path) -> None:
    x = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1, 0, 1], "b": [1, 1, 0, 0, 1, 1, 0, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    model = RandomForestClassifier(n_estimators=5, random_state=1).fit(x, y)
    result = report(model, x, y, output=str(tmp_path / "model.html"))
    assert "brier" in result
    assert "per_class_errors" in result
    assert set(result["per_class_errors"]) == {"0", "1"}


def test_regression_adjusted_r2(tmp_path: Path) -> None:
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(3)
    x = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    y = 2 * x["a"] - x["b"] + rng.normal(scale=0.1, size=40)
    model = LinearRegression().fit(x, y)
    result = report(model, x, y, output=str(tmp_path / "reg.html"))
    assert "adjusted_r2" in result
    assert "explained_variance" in result
    assert result["adjusted_r2"] <= result["r2"] + 1e-9


def test_plugin_registry_and_loader() -> None:
    from fitcheck.plugins import load_plugin, registry

    def dummy_check(df: pd.DataFrame) -> list[dict[str, object]]:
        return []

    registry.register("dummy_check", dummy_check)
    try:
        assert registry.get("dummy_check") is dummy_check
        assert "dummy_check" in registry.list()
        assert load_plugin("dummy_check") is dummy_check
        with pytest.raises(KeyError):
            registry.get("missing_check")
        with pytest.raises(TypeError):
            registry.register("not_callable", 42)
    finally:
        registry.unregister("dummy_check")


def test_load_plugin_dotted_module(tmp_path: Path, monkeypatch) -> None:
    import sys

    from fitcheck.plugins import load_plugin

    (tmp_path / "upg_checkmod.py").write_text("def check(df):\n    return []\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "upg_checkmod", raising=False)  # force a fresh import
    plugin = load_plugin("upg_checkmod")
    assert callable(plugin)
    assert plugin(pd.DataFrame()) == []


def test_full_without_model_writes_index(tmp_path: Path) -> None:
    from fitcheck.cli import main

    data_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [0, 1, 0, 1], "target": [0, 1, 0, 1]}).to_csv(data_path, index=False)
    out_dir = tmp_path / "reports"
    assert main(["full", str(data_path), "--target", "target", "--output-dir", str(out_dir), "--quiet"]) == 0
    assert (out_dir / "index.html").exists()
    assert (out_dir / "dataset_report.html").exists()
    assert not (out_dir / "model_report.html").exists()
    assert "Executive Report" in (out_dir / "index.html").read_text()


def test_exit_code_helper() -> None:
    from fitcheck.cli import _exit_code

    assert _exit_code([], None) == 0
    assert _exit_code([{"severity": "info"}], None) == 0
    assert _exit_code([{"severity": "warning"}], None) == 1
    assert _exit_code([{"severity": "critical"}], None) == 2
    assert _exit_code([{"severity": "warning"}], "critical") == 0
    assert _exit_code([{"severity": "critical"}], "critical") == 2
    assert _exit_code([{"severity": "warning"}], "warning") == 1


def test_magic_parses_and_renders(tmp_path: Path) -> None:
    from fitcheck.magic import _parse_args, _render_inline, _resolve_data, load_ipython_extension

    assert _parse_args("df --target label") == ("df", "label")
    assert _parse_args("--target label") == (None, "label")
    ns = {"df": pd.DataFrame({"a": [1, 2, 3]})}
    assert _resolve_data("df", ns) is ns["df"]
    with pytest.raises(NameError):
        _resolve_data("missing", ns)

    rendered: list[object] = []
    _render_inline(ns["df"], None, rendered.append)
    assert len(rendered) == 1  # an HTML object was produced

    class FakeIP:
        def __init__(self) -> None:
            self.registered: list[tuple[str, str]] = []

        def register_magic_function(self, fn, magic_kind: str, magic_name: str) -> None:
            self.registered.append((magic_kind, magic_name))

    ip = FakeIP()
    load_ipython_extension(ip)
    assert ("line", "fitcheck") in ip.registered
    assert ("cell", "fitcheck") in ip.registered


def test_magic_entry_points_smoke() -> None:
    """Line and cell magics render inline without raising outside a kernel."""
    from fitcheck.magic import _user_ns, run_fitcheck_cell, run_fitcheck_line

    assert isinstance(_user_ns(), dict)
    ns = {"df": pd.DataFrame({"a": [1, 2, 3]})}
    run_fitcheck_line("df", ns=ns)
    run_fitcheck_cell("--target label", "", ns=ns)
    with pytest.raises(NameError):
        run_fitcheck_line("missing_df", ns=ns)
