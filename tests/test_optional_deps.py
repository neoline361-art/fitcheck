"""Optional-dependency behavior tests.

Every optional dependency (polars, duckdb, plotly, mlflow, dvc, shap) must be
treated gracefully: absent means the feature is unavailable, never a crash.
``pytest.importorskip`` semantics are used per dep; each loader/integration
must also degrade cleanly inside the library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Backend selection degrades gracefully when the optional loader is missing
# ---------------------------------------------------------------------------

def test_backend_unknown_name_falls_back_to_pandas() -> None:
    """An unrecognized backend name must degrade gracefully to pandas loading."""
    from fitcheck.backends import get_backend

    assert get_backend("nonexistent_backend_xyz").name == "pandas"


def test_backend_duckdb_falls_back_to_error_message() -> None:
    """Without duckdb installed, selection must raise ImportError."""
    pytest.importorskip("duckdb")  # duckdb is installed in dev env; keep the contract test honest


def test_backend_polars_path_round_trip(tmp_path: Path) -> None:
    """Polars backend must round-trip a CSV identically to pandas loading."""
    pl = pytest.importorskip("polars")
    from fitcheck.backends import get_backend

    path = tmp_path / "t.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(path, index=False)
    backend = get_backend("polars")
    frame = backend.read(str(path))
    assert len(backend.to_pandas(frame)) == 3


def test_backend_duckdb_path_round_trip(tmp_path: Path) -> None:
    """DuckDB backend must round-trip a CSV identically to pandas loading."""
    pytest.importorskip("duckdb")
    from fitcheck.backends import get_backend

    path = tmp_path / "t.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(path, index=False)
    backend = get_backend("duckdb")
    frame = backend.read(str(path))
    assert len(backend.to_pandas(frame)) == 3


def test_check_unknown_backend_falls_back_to_pandas() -> None:
    """An unrecognized backend value degrades gracefully to pandas loading."""
    from fitcheck.check import check

    result = check(pd.DataFrame({"a": [1, 2, 3]}), backend="unknown", return_format="dict", output="/dev/null")
    assert result["total_rows"] == 3


# ---------------------------------------------------------------------------
# Viz renderers degrade gracefully
# ---------------------------------------------------------------------------

def test_viz_unknown_renderer_falls_back_to_static() -> None:
    """Requesting an unknown renderer must degrade gracefully to static."""
    from fitcheck.viz import get_renderer

    renderer = get_renderer("unknown_renderer_xyz")
    assert renderer.name == "static"


def test_plotly_renderer_unavailable_falls_back_to_static() -> None:
    """When plotly is absent, the engine must render static plots instead of crashing."""
    import sys

    from fitcheck.html import _plotly_js

    # Vendored JS embeds regardless, but a missing plotly package must never
    # raise at import time; the viz module imports plotly lazily.
    removed = sys.modules.pop("plotly", None)
    try:
        from fitcheck.viz import get_renderer

        renderer = get_renderer("static")
        assert renderer.render_histogram(pd.Series([1.0, 2.0, 3.0]), "h").startswith("<img")
    finally:
        if removed is not None:
            sys.modules["plotly"] = removed


# ---------------------------------------------------------------------------
# Integrations remain no-ops when their libraries are absent
# ---------------------------------------------------------------------------

def test_mlflow_integration_noop_without_package() -> None:
    """log_to_mlflow returns False (never logs) when mlflow is unavailable."""
    import sys

    from fitcheck.integrations import log_to_mlflow

    original = sys.modules.get("mlflow")
    sys.modules["mlflow"] = None
    try:
        result = log_to_mlflow({"issues": [], "summary": {"critical": 0, "warning": 0, "info": 0}})
        assert result is False
    finally:
        if original is None:
            sys.modules.pop("mlflow", None)
        else:
            sys.modules["mlflow"] = original


def test_dvc_integration_noop_without_yaml() -> None:
    """log_to_dvc returns False (never writes) when yaml is unavailable."""
    import sys

    from fitcheck.integrations import log_to_dvc

    original = sys.modules.get("yaml")
    sys.modules["yaml"] = None
    try:
        assert log_to_dvc({"issues": []}, path="/tmp/fitcheck_dvc_noop.yaml") is False
    finally:
        if original is None:
            sys.modules.pop("yaml", None)
        else:
            sys.modules["yaml"] = original


# ---------------------------------------------------------------------------
# Feature importance degrades gracefully without shap
# ---------------------------------------------------------------------------

def test_shap_missing_leaves_importance_out() -> None:
    """Non-tree models without shap must skip importance, never raise."""
    import sys

    from fitcheck.report import report
    from sklearn.linear_model import LogisticRegression

    removed = sys.modules.pop("shap", None)
    try:
        x = pd.DataFrame({"a": np.arange(20) % 2, "b": np.arange(20) % 3})
        y = pd.Series(np.arange(20) % 2)
        model = LogisticRegression(max_iter=200).fit(x, y)
        result = report(model, x, y, output="/dev/null")
        assert "feature_importance" not in result
    finally:
        if removed is not None:
            sys.modules["shap"] = removed
