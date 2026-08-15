from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn import metrics as sk_metrics
from sklearn.calibration import calibration_curve

from fitcheck.html import render_report_html


def report(
    model: Any,
    x_test: pd.DataFrame | NDArray[Any],
    y_test: pd.Series | NDArray[Any],
    output: str = "model_report.html",
) -> dict[str, Any]:
    """Evaluate a trained model and generate a self-contained HTML report."""
    y_values = _to_array(y_test)
    task = _detect_task(y_values)
    prediction_x: Any = x_test  # preserve DataFrame feature names for sklearn pipelines
    if task == "classification":
        metrics, plots = _classification_report(model, prediction_x, y_values)
    else:
        metrics, plots = _regression_report(model, prediction_x, y_values)
    feature_names = list(x_test.columns) if isinstance(x_test, pd.DataFrame) else None
    importance = _tree_importance(model, _to_array(x_test), feature_names)
    if importance:
        metrics["feature_importance"] = importance
    render_report_html(metrics, plots, task, output)
    return metrics


def _detect_task(y_test: NDArray[Any]) -> str:
    unique = np.unique(y_test)
    if len(unique) <= 2 or (y_test.dtype.kind in "iOb" and len(unique) <= 20):
        return "classification"
    return "regression"


def _to_array(data: pd.DataFrame | pd.Series | NDArray[Any]) -> NDArray[Any]:
    return np.asarray(data.values) if hasattr(data, "values") else np.asarray(data)


def _classification_report(
    model: Any, x_test: Any, y_test: NDArray[Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    y_pred = np.asarray(model.predict(x_test))
    labels = np.unique(y_test)
    metrics: dict[str, Any] = {
        "accuracy": round(float(sk_metrics.accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(sk_metrics.precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(sk_metrics.recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1": round(float(sk_metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "support": int(len(y_test)),
    }
    plots: dict[str, str] = {}
    if len(labels) <= 10:
        cm = sk_metrics.confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Confusion Matrix")
        ax.set_xticks(range(len(labels)), labels=labels)
        ax.set_yticks(range(len(labels)), labels=labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plots["confusion_matrix"] = _fig_to_base64(fig)
        plt.close(fig)
    if len(labels) == 2 and hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x_test))[:, 1]
        positive = labels[1]
        binary_y = (y_test == positive).astype(int)
        fpr, tpr, _ = sk_metrics.roc_curve(binary_y, probabilities)
        roc_auc = sk_metrics.roc_auc_score(binary_y, probabilities)
        precision_curve, recall_curve, thresholds = sk_metrics.precision_recall_curve(binary_y, probabilities)
        metrics["roc_auc"] = round(float(roc_auc), 4)
        metrics["average_precision"] = round(float(sk_metrics.average_precision_score(binary_y, probabilities)), 4)
        metrics["brier"] = round(float(sk_metrics.brier_score_loss(binary_y, probabilities)), 4)
        best_idx = int(np.argmax(2 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 1e-12)))
        metrics["recommended_threshold"] = round(float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) else 0.5, 4)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        plots["roc_curve"] = _fig_to_base64(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall_curve, precision_curve, label=f"AP = {metrics['average_precision']:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve")
        ax.legend()
        plots["pr_curve"] = _fig_to_base64(fig)
        plt.close(fig)
        try:
            frac_pos, mean_pred = calibration_curve(binary_y, probabilities, n_bins=10)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(mean_pred, frac_pos, marker="o", label="Model")
            ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title("Calibration Curve")
            ax.legend()
            plots["calibration_curve"] = _fig_to_base64(fig)
            plt.close(fig)
        except ValueError:
            pass  # too few samples for binning; calibration plot skipped
    metrics["per_class_errors"] = _per_class_errors(y_test, y_pred, labels)
    return metrics, plots


def _per_class_errors(y_test: NDArray[Any], y_pred: NDArray[Any], labels: NDArray[Any]) -> dict[str, float]:
    """Error rate per true class, useful for finding systematically misclassified groups."""
    errors: dict[str, float] = {}
    for label in labels:
        mask = y_test == label
        if not int(mask.sum()):
            continue
        errors[str(label)] = round(float((y_pred[mask] != label).mean()), 4)
    return errors


def _regression_report(model: Any, x_test: Any, y_test: NDArray[Any]) -> tuple[dict[str, Any], dict[str, str]]:
    y_pred = np.asarray(model.predict(x_test))
    mse = float(sk_metrics.mean_squared_error(y_test, y_pred))
    r2 = float(sk_metrics.r2_score(y_test, y_pred))
    n = int(len(y_test))
    p = int(_feature_count(x_test))
    adjusted_r2: float | None = None
    if n - p - 1 > 0:
        adjusted_r2 = round(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1), 4)
    metrics: dict[str, Any] = {
        "mse": round(mse, 4), "rmse": round(float(np.sqrt(mse)), 4),
        "mae": round(float(sk_metrics.mean_absolute_error(y_test, y_pred)), 4),
        "r2": round(r2, 4), "adjusted_r2": adjusted_r2,
        "explained_variance": round(float(sk_metrics.explained_variance_score(y_test, y_pred)), 4),
        "support": n,
    }
    plots: dict[str, str] = {}
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="none")
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    plots["residuals"] = _fig_to_base64(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="none")
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    plots["actual_vs_predicted"] = _fig_to_base64(fig)
    plt.close(fig)
    return metrics, plots


def _feature_count(x_test: Any) -> int:
    """Number of model features, whether X is a DataFrame or an array."""
    if hasattr(x_test, "columns"):
        return int(len(x_test.columns))
    arr = np.asarray(x_test)
    return int(arr.shape[1]) if arr.ndim > 1 else 1


def _tree_importance(model: Any, x_test: NDArray[Any], feature_names: list[str] | None = None) -> dict[str, float] | None:
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    names = feature_names or [f"feature_{i}" for i in range(len(importances))]
    top = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:15]
    return {name: round(float(imp), 4) for name, imp in top}


def _fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
