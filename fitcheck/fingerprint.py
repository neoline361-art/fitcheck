"""Lightweight dataset and report fingerprinting for tamper-evident reports.

Every report embeds a visible footer with:
  - dataset SHA-256 (raw file bytes when available, DataFrame hash otherwise)
  - config hash
  - FitCheck version
  - timestamp
  - optional HMAC-SHA256 signature

Users can call ``verify_report(html_path, csv_path)`` to confirm a report
matches its source data, or use the CLI: ``fitcheck verify report.html --against data.csv``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from datetime import datetime, timezone
from html import escape as _html_escape
from pathlib import Path
from typing import Any

import pandas as pd

from fitcheck._version import __version__


def fingerprint(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    raw_hash: str | None = None,
    result_summary: str = "",
    secret_key: str | None = None,
) -> dict[str, str]:
    """Return a dict of fingerprints for the dataset and configuration."""
    dataset_hash = raw_hash if raw_hash else _hash_dataframe(df)
    fp: dict[str, str] = {
        "dataset_hash": dataset_hash,
        "config_hash": _hash_dict(config),
        "fitcheck_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if secret_key:
        fp["result_summary"] = result_summary
        payload = f"{fp['dataset_hash']}|{fp['config_hash']}|{fp['fitcheck_version']}|{fp['timestamp']}|{result_summary}"
        fp["signature"] = hmac.new(
            secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    return fp


def fingerprint_html(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    raw_hash: str | None = None,
    result_summary: str = "",
    secret_key: str | None = None,
) -> str:
    """Return an HTML fragment to embed in report footers."""
    fp = fingerprint(
        df, config,
        raw_hash=raw_hash,
        result_summary=result_summary,
        secret_key=secret_key,
    )
    sig_html = ""
    if "signature" in fp:
        sig_html = f' · <code>sig: {fp["signature"][:12]}…</code>'
    # Use html.escape for safe attribute embedding (handles &, <, >, ", ')
    return (
        '<div class="fingerprint">'
        f'FitCheck <code>v{fp["fitcheck_version"]}</code> · '
        f'dataset <code>{fp["dataset_hash"][:16]}…</code> · '
        f'config <code>{fp["config_hash"][:16]}…</code> · '
        f'<code>{fp["timestamp"][:19]}</code>'
        f'{sig_html}'
        ' <span style="color:var(--muted);margin-left:6px">'
        f'<code>verify: fitcheck verify --against &lt;data.csv&gt;</code></span>'
        f'<input type="hidden" class="fc-fingerprint" '
        f'value="{_html_escape(json.dumps(fp))}">'
        '</div>'
    )


def hash_file(path: str | Path) -> str:
    """Compute a deterministic SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_report(
    html_path: str | Path,
    csv_path: str | Path | None = None,
    *,
    secret_key: str | None = None,
) -> dict[str, Any]:
    """Check whether an HTML report matches the current source data."""
    html = Path(html_path).read_text(encoding="utf-8")
    match = re.search(
        r'<input[^>]*class="fc-fingerprint"[^>]*value="([^"]+)"', html
    )
    if not match:
        return {"match": False, "message": "No fingerprint found in report"}

    raw_value = match.group(1).replace("&quot;", '"')
    fp = json.loads(raw_value)
    report_hash = fp.get("dataset_hash", "")
    report_version = fp.get("fitcheck_version", "unknown")
    report_timestamp = fp.get("timestamp", "unknown")

    result: dict[str, Any] = {
        "match": False,
        "report_hash": report_hash,
        "current_hash": None,
        "report_version": report_version,
        "report_timestamp": report_timestamp,
        "signature_valid": None,
        "message": "",
    }

    if csv_path is not None:
        csv_path = Path(csv_path)
        current_hash = hash_file(csv_path)
        result["current_hash"] = current_hash
        result["match"] = report_hash == current_hash
    else:
        result["match"] = True
        result["message"] = "Fingerprint present (no source file provided for comparison)"

    # Verify HMAC signature when provided
    if secret_key and "signature" in fp:
        payload = f"{fp['dataset_hash']}|{fp['config_hash']}|{fp['fitcheck_version']}|{fp['timestamp']}|{fp.get('result_summary', '')}"
        expected_sig = hmac.new(
            secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        result["signature_valid"] = hmac.compare_digest(
            fp["signature"], expected_sig
        )
        if not result["signature_valid"]:
            result["match"] = False
            result["message"] = "HMAC signature mismatch — report may be tampered"
            return result

    if result["match"] and result["message"] == "":
        result["message"] = "Report matches current data"
    elif not result["match"] and result["message"] == "":
        result["message"] = "MISMATCH — report may be tampered or data has changed"

    return result


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 of DataFrame content."""
    h = hashlib.sha256()
    for col in sorted(set(df.columns)):
        h.update(col.encode("utf-8"))
        for val in df[col].tolist():
            h.update(repr(val).encode("utf-8"))
    return h.hexdigest()


def _hash_dict(d: dict[str, Any]) -> str:
    """Deterministic SHA-256 of a config dict."""
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
