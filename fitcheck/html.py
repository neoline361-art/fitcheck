"""HTML report rendering — dark-mode templates for check, report, and drift."""

from __future__ import annotations

from typing import Any

import pandas as pd

# Shared CSS for all reports
_DARK_CSS = """
:root {
  --bg: #0d1117;
  --fg: #c9d1d9;
  --card: #161b22;
  --border: #30363d;
  --accent: #58a6ff;
  --critical: #f85149;
  --warning: #d29922;
  --info: #58a6ff;
  --pass: #3fb950;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.6;
}
.container { max-width: 960px; margin: 0 auto; padding: 40px 20px; }
h1, h2 { color: var(--fg); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
}
.badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.badge-critical { background: rgba(248,81,73,0.15); color: var(--critical); }
.badge-warning { background: rgba(210,153,34,0.15); color: var(--warning); }
.badge-info { background: rgba(88,166,255,0.15); color: var(--info); }
.badge-pass { background: rgba(63,185,80,0.15); color: var(--pass); }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 16px; }
.metric-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; text-align: center; }
.metric-value { font-size: 2rem; font-weight: 700; color: var(--accent); }
.metric-label { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }
.issue-list { list-style: none; padding: 0; }
.issue-item {
  background: var(--card);
  border-left: 4px solid var(--border);
  border-radius: 0 8px 8px 0;
  padding: 14px 18px;
  margin-bottom: 12px;
}
.issue-critical { border-left-color: var(--critical); }
.issue-warning { border-left-color: var(--warning); }
.issue-info { border-left-color: var(--info); }
table { width: 100%; border-collapse: collapse; margin-top: 12px; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: #8b949e; font-weight: 500; font-size: 0.8rem; text-transform: uppercase; }
tr:hover { background: rgba(88,166,255,0.04); }
.plot-img { max-width: 100%; border-radius: 8px; border: 1px solid var(--border); margin: 12px 0; }
.footer { text-align: center; margin-top: 40px; color: #8b949e; font-size: 0.8rem; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def _base_html(title: str, body_content: str) -> str:
    """Wrap body content in a complete HTML document with dark styling."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{_DARK_CSS}</style>
</head>
<body>
<div class="container">
{body_content}
</div>
</body>
</html>"""


def render_check_html(
    issues: list[dict[str, Any]], df: pd.DataFrame, output: str
) -> None:
    """Render dataset health check report to HTML."""
    critical = sum(1 for i in issues if i.get("severity") == "critical")
    warning = sum(1 for i in issues if i.get("severity") == "warning")
    info = sum(1 for i in issues if i.get("severity") == "info")
    status = "PASS" if len(issues) == 0 else "ISSUES FOUND"
    status_class = "badge-pass" if len(issues) == 0 else "badge-warning"

    body_parts = []
    body_parts.append(f"""
<h1>FitCheck Dataset Report</h1>
<div class="card">
  <span class="badge {status_class}">{status}</span>
  <div class="metric-grid" style="margin-top:16px">
    <div class="metric-card">
      <div class="metric-value">{len(df)}</div>
      <div class="metric-label">Rows</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{len(df.columns)}</div>
      <div class="metric-label">Columns</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:var(--critical)">{critical}</div>
      <div class="metric-label">Critical</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:var(--warning)">{warning}</div>
      <div class="metric-label">Warnings</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:var(--info)">{info}</div>
      <div class="metric-label">Info</div>
    </div>
  </div>
</div>
""")

    if issues:
        body_parts.append('<h2>Issues</h2><ul class="issue-list">\n')
        for issue in issues:
            sev = issue.get("severity", "info")
            css_class = f"issue-{sev}"
            body_parts.append(f"""
<li class="issue-item {css_class}">
  <strong>{issue.get("type", "").replace("_", " ").title()}</strong>
  <span class="badge badge-{sev}">{sev}</span><br>
  <code>{issue.get("column", "")}</code> — {issue.get("message", "")}<br>
  <small>Suggestion: {issue.get("suggestion", "")}</small>
</li>
""")
        body_parts.append("</ul>\n")
    else:
        body_parts.append('<div class="card"><p>No issues detected. Dataset looks clean!</p></div>\n')

    # Data preview
    body_parts.append('<h2>Data Preview</h2>\n<div class="card">\n')
    body_parts.append(df.head(10).to_html(index=False, classes="preview-table"))
    body_parts.append("\n</div>\n")

    body_parts.append('<div class="footer">Generated by <a href="https://github.com/neoline361-art/fitcheck">FitCheck v2.0</a></div>\n')

    with open(output, "w", encoding="utf-8") as f:
        f.write(_base_html("FitCheck Dataset Report", "".join(body_parts)))


def render_report_html(
    metrics: dict[str, Any], plots: dict[str, str], task: str, output: str
) -> None:
    """Render model evaluation report to HTML."""
    body_parts = []
    body_parts.append(f"""
<h1>FitCheck Model Report</h1>
<div class="card">
  <span class="badge badge-info">{task.upper()}</span>
</div>
""")

    # Metrics grid
    body_parts.append('<h2>Metrics</h2><div class="metric-grid">\n')
    for key, value in metrics.items():
        if key == "feature_importance":
            continue
        label = key.replace("_", " ").upper()
        if isinstance(value, float):
            display = f"{value:.4f}"
        else:
            display = str(value)
        body_parts.append(f"""
<div class="metric-card">
  <div class="metric-value">{display}</div>
  <div class="metric-label">{label}</div>
</div>
""")
    body_parts.append("</div>\n")

    # Plots
    if plots:
        body_parts.append("<h2>Visualizations</h2>\n")
        for name, b64 in plots.items():
            title = name.replace("_", " ").title()
            body_parts.append(f"""
<div class="card">
  <h3>{title}</h3>
  <img class="plot-img" src="data:image/png;base64,{b64}" alt="{title}">
</div>
""")

    # Feature importance
    if "feature_importance" in metrics:
        body_parts.append('<h2>Feature Importance (Top 15)</h2>\n<div class="card">\n<table>\n<tr><th>Feature</th><th>Importance</th></tr>\n')
        for feat, imp in list(metrics["feature_importance"].items())[:15]:
            body_parts.append(f"<tr><td>{feat}</td><td>{imp:.4f}</td></tr>\n")
        body_parts.append("</table>\n</div>\n")

    body_parts.append('<div class="footer">Generated by <a href="https://github.com/neoline361-art/fitcheck">FitCheck v2.0</a></div>\n')

    with open(output, "w", encoding="utf-8") as f:
        f.write(_base_html("FitCheck Model Report", "".join(body_parts)))


def render_drift_html(
    results: list[dict[str, Any]],
    ref_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    output: str,
) -> None:
    """Render drift detection report to HTML."""
    drifted = sum(1 for r in results if r.get("drifted"))
    status = "DRIFT DETECTED" if drifted > 0 else "NO DRIFT"
    status_class = "badge-critical" if drifted > 0 else "badge-pass"

    body_parts = []
    body_parts.append(f"""
<h1>FitCheck Drift Report</h1>
<div class="card">
  <span class="badge {status_class}">{status}</span>
  <div class="metric-grid" style="margin-top:16px">
    <div class="metric-card">
      <div class="metric-value">{len(ref_df)}</div>
      <div class="metric-label">Reference Rows</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{len(prod_df)}</div>
      <div class="metric-label">Production Rows</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{len(results)}</div>
      <div class="metric-label">Features Tested</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:var(--critical)">{drifted}</div>
      <div class="metric-label">Drifted</div>
    </div>
  </div>
</div>
""")

    body_parts.append('<h2>Per-Feature Results</h2>\n<div class="card">\n<table>\n')
    body_parts.append("<tr><th>Feature</th><th>Type</th><th>Test</th><th>Statistic</th><th>P-Value</th><th>Result</th></tr>\n")
    for r in results:
        r.get("severity", "info")
        result_text = "DRIFT" if r.get("drifted") else "OK"
        result_class = "badge-critical" if r.get("drifted") else "badge-pass"
        body_parts.append(f"""
<tr>
  <td>{r.get("feature", "")}</td>
  <td>{r.get("type", "")}</td>
  <td>{r.get("test", "")}</td>
  <td>{r.get("statistic", "")}</td>
  <td>{r.get("p_value", "")}</td>
  <td><span class="badge {result_class}">{result_text}</span></td>
</tr>
""")
    body_parts.append("</table>\n</div>\n")

    # Messages
    if drifted > 0:
        body_parts.append('<h2>Details</h2>\n<ul class="issue-list">\n')
        for r in results:
            if r.get("drifted"):
                body_parts.append(f"""
<li class="issue-item issue-critical">
  <strong>{r.get("feature", "")}</strong> — {r.get("message", "")}
</li>
""")
        body_parts.append("</ul>\n")

    body_parts.append('<div class="footer">Generated by <a href="https://github.com/neoline361-art/fitcheck">FitCheck v2.0</a></div>\n')

    with open(output, "w", encoding="utf-8") as f:
        f.write(_base_html("FitCheck Drift Report", "".join(body_parts)))
