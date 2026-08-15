"""Visualization renderers: static matplotlib (default) and optional Plotly."""

from __future__ import annotations

from typing import Any


class BaseVizRenderer:
    """Abstract interface for chart renderers.

    Every method returns an HTML fragment: an ``<img>`` tag (static renderer)
    or a Plotly ``<div>`` + inline script (plotly renderer).
    """

    name: str = "base"

    def render_histogram(self, series: Any, title: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def render_roc_curve(self, fpr: Any, tpr: Any, auc: float) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def render_confusion_matrix(self, cm: Any, labels: list[str]) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def render_feature_importance(self, features: list[str], importance: list[float]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


def get_renderer(name: str = "static") -> BaseVizRenderer:
    """Return the named renderer; unknown names fall back to static."""
    if name == "plotly":
        try:
            from fitcheck.viz.plotly import PlotlyRenderer

            return PlotlyRenderer()
        except ImportError:  # pragma: no cover - depends on optional package
            pass
    return StaticRenderer()


class StaticRenderer(BaseVizRenderer):
    """Matplotlib renderer emitting base64 ``<img>`` tags (offline-safe)."""

    name = "static"

    def render_histogram(self, series: Any, title: str) -> str:
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series.dropna(), bins=30, color="#60a5fa")
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f'<img class="plot-img" src="data:image/png;base64,{b64}" alt="{title}">'

    def render_roc_curve(self, fpr: Any, tpr: Any, auc: float) -> str:
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color="#4ade80", label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f'<img class="plot-img" src="data:image/png;base64,{b64}" alt="ROC curve">'

    def render_confusion_matrix(self, cm: Any, labels: list[str]) -> str:
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(np.asarray(cm), cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(labels)), labels=labels)
        ax.set_yticks(range(len(labels)), labels=labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f'<img class="plot-img" src="data:image/png;base64,{b64}" alt="Confusion matrix">'

    def render_feature_importance(self, features: list[str], importance: list[float]) -> str:
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(features, importance, color="#7dd3fc")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        ax.invert_yaxis()
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f'<img class="plot-img" src="data:image/png;base64,{b64}" alt="Feature importance">'
