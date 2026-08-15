"""Plotly renderer producing interactive, self-contained HTML fragments.

Requires the optional ``plotly`` package. The vendored ``plotly.min.js`` is
embedded once in the report head so charts work offline (no CDN dependency).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from fitcheck.viz import BaseVizRenderer


@lru_cache(maxsize=1)
def plotly_js() -> str:
    """Return the vendored Plotly JavaScript for embedding in a report head."""
    asset = Path(__file__).parent / "plotly.min.js"
    return asset.read_text(encoding="utf-8") if asset.exists() else ""


class PlotlyRenderer(BaseVizRenderer):
    """Interactive renderer emitting Plotly divs plus inline scripts."""

    name = "plotly"

    def __init__(self) -> None:
        try:
            import plotly.graph_objects as go  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "The plotly renderer requires the 'plotly' package; install it with "
                "pip install data-fitcheck[plotly]"
            ) from exc
        self._go = go

    def _div(self, fig: Any) -> str:
        import json

        from plotly.utils import PlotlyJSONEncoder

        payload = json.dumps(fig, cls=PlotlyJSONEncoder)
        div_id = f"plotly-{abs(hash(payload)) % (10**9)}"
        return (
            f'<div id="{div_id}" class="plotly-div" style="width:100%;height:340px"></div>'
            f'<script>Plotly.newPlot("{div_id}", {payload}.data, {payload}.layout, '
            f"{{responsive:true}});</script>"
        )

    def render_histogram(self, series: Any, title: str) -> str:
        fig = self._go.Figure(
            data=[
                self._go.Histogram(
                    x=series.dropna().tolist(),
                    nbinsx=30,
                    marker_color="#60a5fa",
                    opacity=0.85,
                )
            ]
        )
        fig.update_layout(title=title, template="plotly_dark", height=340)
        return self._div(fig)

    def render_roc_curve(self, fpr: Any, tpr: Any, auc: float) -> str:
        fig = self._go.Figure()
        fig.add_trace(
            self._go.Scatter(
                x=list(fpr), y=list(tpr), mode="lines", name=f"AUC = {auc:.3f}",
                line=dict(color="#4ade80", width=2),
            )
        )
        fig.add_trace(
            self._go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="#666", dash="dash"))
        )
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate", template="plotly_dark", height=340)
        return self._div(fig)

    def render_confusion_matrix(self, cm: Any, labels: list[str]) -> str:
        matrix = np.asarray(cm).tolist()
        fig = self._go.Figure(
            data=self._go.Heatmap(
                z=matrix, x=list(labels), y=list(labels),
                colorscale="Viridis", text=matrix, texttemplate="%{text}",
                hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
            )
        )
        fig.update_layout(title="Confusion Matrix", template="plotly_dark", height=380)
        return self._div(fig)

    def render_feature_importance(self, features: list[str], importance: list[float]) -> str:
        fig = self._go.Figure(
            data=self._go.Bar(
                x=list(importance), y=list(features), orientation="h",
                marker_color="#7dd3fc",
            )
        )
        fig.update_layout(title="Feature Importance", xaxis_title="Importance",
                          template="plotly_dark", height=380)
        return self._div(fig)
