"""Model evaluation engine — auto-detects classification vs regression."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from fitcheck.html import render_report_html


def report(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    output: str = "model_report.html",
) -> Dict[str, Any]:
    """Evaluate a trained model and generate an HTML report.

    Args:
        model: Trained scikit-learn model with a predict method.
        X_test: Test features.
        y_test: Test targets.
        output: Path for the generated HTML report.

    Returns:
        Dictionary containing all computed metrics.
    """
    X_test = _to_array(X_test)
    y_test = _to_array(y_test)

    task = _detect_task(y_test)

    if task == "classification":
        metrics, plots = _classification_report(model, X_test, y_test)
    else:
        metrics, plots = _regression_report(model, X_test, y_test)

    importance = _tree_importance(model, X_test)
    if importance:
        metrics["feature_importance"] = importance

    render_report_html(metrics, plots, task, output)
    return metrics


def _detect_task(y_test: np.ndarray) -> str:
    """Auto-detect if the task is classification or regression."""
    unique = np.unique(y_test)
    if len(unique) <= 2 or (y_test.dtype.kind in "iOb" and len(unique) <= 20):
        return "classification"
    return "regression"


def _to_array(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    """Convert pandas or numpy input to numpy array."""
    if hasattr(data, "values"):
        return data.values
    return np.asarray(data)


def _classification_report(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Generate classification metrics and plots."""
    y_pred = model.predict(X_test)

    metrics: Dict[str, Any] = {
        "accuracy": round(float(sk_metrics.accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(sk_metrics.precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(sk_metrics.recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1": round(float(sk_metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
    }

    plots: Dict[str, str] = {}

    # Confusion matrix heatmap
    if len(np.unique(y_test)) <= 10:
        cm = sk_metrics.confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        plots["confusion_matrix"] = _fig_to_base64(fig)
        plt.close(fig)

    # ROC curve (binary only)
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = sk_metrics.roc_curve(y_test, y_prob)
        roc_auc = sk_metrics.roc_auc_score(y_test, y_prob)
        metrics["roc_auc"] = round(float(roc_auc), 4)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        plt.tight_layout()
        plots["roc_curve"] = _fig_to_base64(fig)
        plt.close(fig)

    return metrics, plots


def _regression_report(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Generate regression metrics and plots."""
    y_pred = model.predict(X_test)

    mse = float(sk_metrics.mean_squared_error(y_test, y_pred))
    metrics: Dict[str, Any] = {
        "mse": round(mse, 4),
        "rmse": round(np.sqrt(mse), 4),
        "mae": round(float(sk_metrics.mean_absolute_error(y_test, y_pred)), 4),
        "r2": round(float(sk_metrics.r2_score(y_test, y_pred)), 4),
    }

    plots: Dict[str, str] = {}

    # Residuals plot
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="none")
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    plt.tight_layout()
    plots["residuals"] = _fig_to_base64(fig)
    plt.close(fig)

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="none")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    plots["actual_vs_predicted"] = _fig_to_base64(fig)
    plt.close(fig)

    return metrics, plots


def _tree_importance(model: Any, X_test: np.ndarray) -> Optional[Dict[str, float]]:
    """Extract feature importances from tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    names = [f"feature_{i}" for i in range(len(importances))]
    top = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:15]
    return {name: round(float(imp), 4) for name, imp in top}


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
