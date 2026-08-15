"""Tests for the Week 2 / v3.1 additions: new checks, backends, renderers, integrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fitcheck.check import check
from fitcheck.extensions import detect_seasonality
from fitcheck.report import report


def test_text_encoding_check_detected(tmp_path: Path) -> None:
    data = pd.DataFrame({"clean": ["ok", "fine"], "bad": ["a", "b\udcffc"]})  # lone surrogate
    result = check(data, output=str(tmp_path / "enc.html"), return_format="dict")
    assert any(issue["type"] == "text_encoding" and issue["column"] == "bad" for issue in result["issues"])


def test_text_encoding_check_clean_columns_pass(tmp_path: Path) -> None:
    data = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["Paris", "Berlin"]})
    result = check(data, output=str(tmp_path / "enc.html"), return_format="dict")
    assert not any(issue["type"] == "text_encoding" for issue in result["issues"])


def test_detect_seasonality_weekly_pattern() -> None:
    rng = np.random.default_rng(7)
    t = np.arange(120)
    series = pd.Series(np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.1, len(t)), name="sales")
    issue = detect_seasonality(series, period=7)
    assert issue is not None
    assert issue["type"] == "timeseries_seasonality"
    assert issue["severity"] == "info"


def test_detect_seasonality_no_pattern_returns_none() -> None:
    rng = np.random.default_rng(8)
    series = pd.Series(rng.normal(0, 1, 100), name="noise")
    assert detect_seasonality(series) is None


def test_detect_seasonality_short_series_returns_none() -> None:
    assert detect_seasonality(pd.Series([1.0, 2.0, 3.0])) is None


def test_shap_fallback_returns_none_without_shap(tmp_path: Path) -> None:
    """Non-tree models without shap installed yield no importance (no crash)."""
    class DummyModel:
        def predict(self, x):
            return np.zeros(len(np.asarray(x)))

    x = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})
    y = pd.Series([0, 0, 1, 1])
    result = report(DummyModel(), x, y, output=str(tmp_path / "model.html"))
    assert "feature_importance" not in result  # shap absent -> no importance key


def test_tree_importance_still_works(tmp_path: Path) -> None:
    x = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1], "b": [1, 1, 0, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = RandomForestClassifier(n_estimators=5, random_state=1).fit(x, y)
    result = report(model, x, y, output=str(tmp_path / "model.html"))
    assert "feature_importance" in result
    assert set(result["feature_importance"]) == {"a", "b"}


def test_get_backend_defaults_and_selection(tmp_path: Path) -> None:
    from fitcheck.backends import get_backend

    assert get_backend().name == "pandas"
    assert get_backend("pandas").name == "pandas"
    # polars is a dev dependency, so the explicit request must succeed.
    pl = pytest.importorskip("polars")
    assert get_backend("polars").name == "polars"
    # Auto-selection: a polars DataFrame passed without an explicit name.
    assert get_backend(df=pl.DataFrame({"a": [1]})).name == "polars"
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    frame = get_backend("polars").read(path.as_posix())
    assert len(frame.to_pandas()) == 3


def test_check_backend_pandas_path(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    result = check(path.as_posix(), output=str(tmp_path / "c.html"), return_format="dict", backend="pandas")
    assert result["total_rows"] == 3


def test_check_backend_polars_path(tmp_path: Path) -> None:
    """End-to-end: the polars backend loads a file and the check engine runs on pandas."""
    pytest.importorskip("polars")
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    result = check(path.as_posix(), output=str(tmp_path / "c.html"), return_format="dict", backend="polars")
    assert result["total_rows"] == 3


def test_plotly_renderer_requires_package() -> None:
    from fitcheck.viz import get_renderer

    # plotly is installed in dev extras; the renderer must construct and emit a div.
    pytest.importorskip("plotly")
    renderer = get_renderer("plotly")
    assert renderer.name == "plotly"
    assert renderer.render_histogram(pd.Series([1.0, 2.0, 3.0]), "h").startswith("<div")
    assert renderer.render_feature_importance(["a"], [0.5]).startswith("<div")


def test_static_renderer_fragments() -> None:
    from fitcheck.viz import get_renderer

    renderer = get_renderer("static")
    assert renderer.name == "static"
    assert renderer.render_histogram(pd.Series([1.0, 2.0, 3.0]), "hist").startswith("<img")
    assert renderer.render_roc_curve([0, 0.5, 1], [0, 0.8, 1], 0.9).startswith("<img")
    assert renderer.render_confusion_matrix(np.array([[2, 0], [0, 2]]), ["0", "1"]).startswith("<img")
    assert renderer.render_feature_importance(["a"], [0.5]).startswith("<img")


def test_report_plotly_renderer_embeds_script(tmp_path: Path) -> None:
    x = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1, 0, 1], "b": [1, 1, 0, 0, 1, 1, 0, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    model = RandomForestClassifier(n_estimators=5, random_state=1).fit(x, y)
    out = tmp_path / "plotly.html"
    report(model, x, y, output=str(out), renderer="plotly")
    html = out.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "plotly-div" in html


def test_integrations_noop_without_deps(tmp_path: Path, monkeypatch) -> None:
    import sys

    from fitcheck.integrations import log_to_dvc, log_to_mlflow

    monkeypatch.setitem(sys.modules, "mlflow", None)
    monkeypatch.setitem(sys.modules, "yaml", None)
    result = {"issues": [], "summary": {"critical": 0, "warning": 0, "info": 0}}
    assert log_to_mlflow(result) is False
    assert log_to_dvc(result, path=str(tmp_path / "m.yaml")) is False
    assert not (tmp_path / "m.yaml").exists()


def test_demo_no_browser_flag(tmp_path: Path, capsys) -> None:
    from fitcheck.cli import main

    assert main(["demo", "--no-browser", "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "demo_check_report.html").exists()
    assert (tmp_path / "demo_model_report.html").exists()
    assert (tmp_path / "demo_drift_report.html").exists()
    assert "Demo complete" in capsys.readouterr().out
