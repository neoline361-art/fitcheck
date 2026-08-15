from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

_DARK_CSS = """
:root{--bg:#0b1020;--fg:#e6edf7;--muted:#91a0b8;--card:#121a2b;--card2:#182238;--border:#293754;--accent:#7dd3fc;--critical:#fb7185;--warning:#fbbf24;--info:#60a5fa;--pass:#4ade80}
*{box-sizing:border-box}body{margin:0;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:linear-gradient(135deg,#0b1020,#101a31);color:var(--fg);line-height:1.55}.container{max-width:1180px;margin:0 auto;padding:40px 22px}h1{font-size:2.35rem;letter-spacing:-.04em;margin:0 0 8px}h2{font-size:1.2rem;margin:30px 0 12px;border-bottom:1px solid var(--border);padding-bottom:8px}h3{margin-top:0}.subtitle{color:var(--muted);margin:0 0 24px}.card,.metric-card{background:rgba(18,26,43,.92);border:1px solid var(--border);border-radius:14px;box-shadow:0 10px 35px rgba(0,0,0,.18)}.card{padding:20px;margin-bottom:20px}.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:14px}.metric-card{padding:16px;text-align:center}.metric-value{font-size:1.8rem;font-weight:750;color:var(--accent);word-break:break-word}.metric-label{font-size:.72rem;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.07em}.badge{display:inline-block;padding:4px 11px;border-radius:999px;font-size:.72rem;font-weight:750;text-transform:uppercase;letter-spacing:.06em}.badge-critical{background:rgba(251,113,133,.14);color:var(--critical)}.badge-warning{background:rgba(251,191,36,.14);color:var(--warning)}.badge-info{background:rgba(96,165,250,.14);color:var(--info)}.badge-pass{background:rgba(74,222,128,.14);color:var(--pass)}.issue-list{list-style:none;padding:0}.issue-item{background:var(--card);border-left:4px solid var(--border);border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:12px}.issue-critical{border-left-color:var(--critical)}.issue-warning{border-left-color:var(--warning)}.issue-info{border-left-color:var(--info)}code{background:var(--card2);padding:2px 6px;border-radius:5px;color:var(--accent)}table{width:100%;border-collapse:collapse;margin-top:4px;display:block;overflow-x:auto}th,td{padding:11px 12px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}th{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}tr:hover{background:rgba(125,211,252,.05)}.plot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}.plot-img{max-width:100%;border-radius:9px;border:1px solid var(--border);margin-top:10px;background:#fff}.footer{text-align:center;margin-top:44px;color:var(--muted);font-size:.8rem}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.callout{border-left:4px solid var(--accent);padding:12px 16px;background:rgba(96,165,250,.08);border-radius:0 8px 8px 0}
"""


def _base_html(title: str, body_content: str, head_extra: str = "") -> str:
    return f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><meta name="generator" content="FitCheck"><title>{escape(title)}</title>{head_extra}<style>{_DARK_CSS}</style></head><body><main class="container">{body_content}</main></body></html>'''


def _plotly_js() -> str:
    """Return the vendored Plotly JS for embedding (empty when the asset is absent)."""
    from pathlib import Path

    asset = Path(__file__).parent / "viz" / "plotly.min.js"
    return asset.read_text(encoding="utf-8") if asset.exists() else ""


def _metric_card(label: str, value: Any, color: str | None = None) -> str:
    style = f' style="color:var(--{color})"' if color else ""
    return f'<div class="metric-card"><div class="metric-value"{style}>{escape(str(value))}</div><div class="metric-label">{escape(label)}</div></div>'


def render_check_html(issues: list[dict[str, Any]], df: pd.DataFrame, output: str | None) -> str:
    """Render the dataset report; writes to ``output`` when provided and returns the HTML."""
    counts = {level: sum(1 for i in issues if i.get("severity") == level) for level in ("critical", "warning", "info")}
    status = "PASS" if not issues else "ISSUES FOUND"
    status_class = "badge-pass" if not issues else "badge-warning"
    parts = [f'<h1>FitCheck Dataset Report</h1><p class="subtitle">A local, read-only health check with actionable next steps.</p><div class="card"><span class="badge {status_class}">{status}</span><div class="metric-grid" style="margin-top:16px">', _metric_card("Rows", len(df)), _metric_card("Columns", len(df.columns)), _metric_card("Critical", counts["critical"], "critical"), _metric_card("Warnings", counts["warning"], "warning"), _metric_card("Info", counts["info"], "info"), '</div></div>']
    if issues:
        parts.append('<h2>Issues and recommendations</h2><ul class="issue-list">')
        for issue in issues:
            severity = str(issue.get("severity", "info"))
            parts.append(f'<li class="issue-item issue-{escape(severity)}"><strong>{escape(str(issue.get("type", "")).replace("_", " ").title())}</strong> <span class="badge badge-{escape(severity)}">{escape(severity)}</span><br><code>{escape(str(issue.get("column", "")))}</code> — {escape(str(issue.get("message", "")))}<br><small>Recommendation: {escape(str(issue.get("suggestion", "")))}</small></li>')
        parts.append("</ul>")
    else:
        parts.append('<div class="callout"><strong>No issues detected.</strong> The dataset passed every configured check.</div>')
    parts.extend(['<h2>Data preview</h2><div class="card">', df.head(10).to_html(index=False, classes="preview-table", escape=True), '</div><div class="footer">Generated locally by <a href="https://github.com/neoline361-art/fitcheck">FitCheck</a></div>'])
    return _render(_base_html("FitCheck Dataset Report", "".join(parts)), output)


def render_report_html(metrics: dict[str, Any], plots: dict[str, str], task: str, output: str | None, renderer: str = "static") -> str:
    """Render the model report; writes to ``output`` when provided and returns the HTML."""
    head_extra = f"<script>{_plotly_js()}</script>" if renderer == "plotly" else ""
    parts = [f'<h1>FitCheck Model Report</h1><p class="subtitle">Evaluation diagnostics for a {escape(task)} task.</p><div class="card"><span class="badge badge-info">{escape(task.upper())}</span></div><h2>Metrics</h2><div class="metric-grid">']
    for key, value in metrics.items():
        if key not in ("feature_importance", "per_class_errors"):
            display = f"{value:.4f}" if isinstance(value, float) else value
            parts.append(_metric_card(key.replace("_", " "), display))
    parts.append("</div>")
    if plots:
        parts.append('<h2>Visualizations</h2><div class="plot-grid">')
        for name, fragment in plots.items():
            title = name.replace("_", " ").title()
            if fragment.startswith("<"):
                parts.append(f'<div class="card"><h3>{escape(title)}</h3>{fragment}</div>')
            else:
                parts.append(f'<div class="card"><h3>{escape(title)}</h3><img class="plot-img" src="data:image/png;base64,{fragment}" alt="{escape(title)}"></div>')
        parts.append("</div>")
    if "per_class_errors" in metrics:
        parts.append('<h2>Per-class error analysis</h2><div class="card"><table><tr><th>Class</th><th>Error rate</th></tr>')
        for label, rate in metrics["per_class_errors"].items():
            parts.append(f"<tr><td>{escape(str(label))}</td><td>{float(rate):.4f}</td></tr>")
        parts.append("</table></div>")
    if "feature_importance" in metrics:
        parts.append('<h2>Feature importance</h2><div class="card"><table><tr><th>Feature</th><th>Importance</th></tr>')
        for feat, imp in list(metrics["feature_importance"].items())[:15]:
            parts.append(f"<tr><td>{escape(str(feat))}</td><td>{float(imp):.4f}</td></tr>")
        parts.append("</table></div>")
    parts.append('<div class="footer">Generated locally by <a href="https://github.com/neoline361-art/fitcheck">FitCheck</a></div>')
    return _render(_base_html("FitCheck Model Report", "".join(parts), head_extra=head_extra), output)


def render_drift_html(results: list[dict[str, Any]], ref_df: pd.DataFrame, prod_df: pd.DataFrame, output: str | None) -> str:
    """Render the drift report; writes to ``output`` when provided and returns the HTML."""
    drifted = sum(1 for r in results if r.get("drifted"))
    status = "DRIFT DETECTED" if drifted else "NO DRIFT"
    status_class = "badge-critical" if drifted else "badge-pass"
    parts = [f'<h1>FitCheck Drift Report</h1><p class="subtitle">Reference-versus-production distribution comparison.</p><div class="card"><span class="badge {status_class}">{status}</span><div class="metric-grid" style="margin-top:16px">', _metric_card("Reference rows", len(ref_df)), _metric_card("Production rows", len(prod_df)), _metric_card("Features tested", len(results)), _metric_card("Drifted", drifted, "critical"), '</div></div><h2>Per-feature results</h2><div class="card"><table><tr><th>Feature</th><th>Type</th><th>Test</th><th>Statistic</th><th>P-value</th><th>Result</th></tr>']
    for result in results:
        drift = bool(result.get("drifted"))
        parts.append(f'<tr><td>{escape(str(result.get("feature", "")))}</td><td>{escape(str(result.get("type", "")))}</td><td>{escape(str(result.get("test", "")))}</td><td>{escape(str(result.get("statistic", "")))}</td><td>{escape(str(result.get("p_value", "")))}</td><td><span class="badge {"badge-critical" if drift else "badge-pass"}">{"DRIFT" if drift else "OK"}</span></td></tr>')
    parts.append("</table></div>")
    if drifted:
        parts.append('<h2>Details</h2><ul class="issue-list">')
        for result in results:
            if result.get("drifted"):
                parts.append(f'<li class="issue-item issue-critical"><strong>{escape(str(result.get("feature", "")))}</strong> — {escape(str(result.get("message", "")))}</li>')
        parts.append("</ul>")
    parts.append('<div class="footer">Generated locally by <a href="https://github.com/neoline361-art/fitcheck">FitCheck</a></div>')
    return _render(_base_html("FitCheck Drift Report", "".join(parts)), output)


def render_full_html(summary: dict[str, Any], output: str) -> str:
    """Render an executive index report for ``fitcheck full``."""
    dataset = summary.get("dataset", {})
    model = summary.get("model", {})
    drift = summary.get("drift", {})
    parts = [
        '<h1>FitCheck Executive Report</h1>',
        '<p class="subtitle">Dataset health, model evaluation, and drift — one overview.</p>',
        '<div class="metric-grid">',
        _metric_card("Dataset issues", dataset.get("issues", 0), "warning" if dataset.get("issues") else "pass"),
        _metric_card("Dataset critical", dataset.get("critical", 0), "critical" if dataset.get("critical") else None),
        _metric_card("Model task", str(model.get("task", "—")).upper()),
        _metric_card("Drift features", drift.get("features", 0)),
        _metric_card("Drifted", drift.get("drifted", 0), "critical" if drift.get("drifted") else "pass"),
        '</div>',
        '<h2>Reports</h2><ul class="issue-list">',
        f'<li class="issue-item"><a href="dataset_report.html"><strong>Dataset health</strong></a> — {dataset.get("issues", 0)} issues ({dataset.get("critical", 0)} critical)</li>',
        f'<li class="issue-item"><a href="model_report.html"><strong>Model evaluation</strong></a> — {model.get("task", "not run")}</li>',
        f'<li class="issue-item"><a href="drift_report.html"><strong>Drift detection</strong></a> — {drift.get("drifted", 0)} of {drift.get("features", 0)} features drifted</li>',
        '</ul>',
        '<div class="footer">Generated locally by <a href="https://github.com/neoline361-art/fitcheck">FitCheck</a></div>',
    ]
    return _render(_base_html("FitCheck Executive Report", "".join(parts)), output)


def _render(content: str, output: str | None) -> str:
    """Write ``content`` to ``output`` when given and always return it."""
    if output is not None:
        _write(output, content)
    return content


def _write(output: str, content: str) -> None:
    # errors="replace" keeps reports writable even when the data preview
    # contains lone surrogates from mixed-encoding inputs.
    with open(output, "w", encoding="utf-8", errors="replace") as file:
        file.write(content)
