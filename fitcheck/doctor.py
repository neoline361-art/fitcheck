"""Environment self-diagnosis: ``fitcheck doctor``.

Checks the Python version, required and optional dependencies, CI-quality
tools, and common configuration problems, then prints an actionable status
report. Exits 0 when every check passes and 2 when at least one critical
problem is found so CI can enforce a healthy development environment.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DoctorCheck:
    """One environment health check with a human-readable result."""

    name: str
    status: str  # "ok", "warning", "critical"
    detail: str


_REQUIRED = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("jinja2", "jinja2"),
    ("filelock", "filelock"),
    ("msgpack", "msgpack"),
    ("requests", "requests"),
    ("urllib3", "urllib3"),
]

_OPTIONAL = {
    "polars": "Polars CSV loading backend (--backend polars)",
    "duckdb": "DuckDB loading backend for very large files (--backend duckdb)",
    "plotly": "Interactive Plotly report renderer (--renderer plotly)",
    "pyarrow": "PyArrow string-backend performance",
    "shap": "Shapley-value feature importance for non-tree models",
    "mlflow": "MLflow run logging (log_to_mlflow)",
    "dvc": "DVC metric logging (log_to_dvc)",
    "yaml": "YAML support for DVC metric logging",
    "ipython": "IPython magics (%fitcheck)",
}

_QUALITY_TOOLS = {
    "pytest": "Test runner",
    "ruff": "Fast linter and formatter",
    "mypy": "Static type checker",
    "bandit": "Security linter",
    "pre_commit": "pre-commit framework (package name: pre-commit)",
}


def _check_python_version() -> DoctorCheck:
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 10:
        return DoctorCheck("python_version", "ok", f"Python {major}.{minor} (>=3.10 supported)")
    if major == 3 and minor >= 9:
        return DoctorCheck("python_version", "warning", f"Python {major}.{minor} works but 3.10+ is recommended")
    return DoctorCheck("python_version", "critical", f"Python {major}.{minor} is not supported (need >=3.9)")


def _check_package(name: str) -> tuple[str, str]:
    """Return (version, status) for an importable package."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        return "missing", "critical"
    version = getattr(module, "__version__", None) or getattr(module, "VERSION", "unknown")
    return str(version), "ok"


def _check_quality_tool(name: str) -> tuple[str, str]:
    """Return (location, status) for a CLI quality tool on PATH."""
    from shutil import which

    path = which(name)
    if path is None:
        return "missing", "warning"
    return str(Path(path)), "ok"


def run_doctor_checks() -> list[DoctorCheck]:
    """Run every environment check and return the results."""
    checks: list[DoctorCheck] = [_check_python_version()]
    for package, import_name in _REQUIRED:
        version, status = _check_package(import_name)
        detail = f"version {version}" if status == "ok" else "required package missing"
        checks.append(DoctorCheck(f"required:{package}", status, detail))
    for package, purpose in _OPTIONAL.items():
        version, status = _check_package(package)
        detail = f"{purpose}: version {version}" if status == "ok" else f"{purpose}: not installed"
        checks.append(DoctorCheck(f"optional:{package}", "warning" if status == "critical" else status, detail))
    for tool, purpose in _QUALITY_TOOLS.items():
        location, status = _check_quality_tool(tool)
        detail = f"{purpose}: {location}" if status == "ok" else f"{purpose}: not on PATH"
        checks.append(DoctorCheck(f"tool:{tool}", status, detail))
    return checks


def format_doctor_report(checks: list[DoctorCheck]) -> str:
    """Render the doctor results as a human-readable report."""
    lines = ["FitCheck environment diagnosis", "=" * 40]
    for check in checks:
        marker = {"ok": "✓", "warning": "⚠", "critical": "✗"}[check.status]
        lines.append(f"{marker} {check.name}: {check.detail}")
    counts = {level: sum(1 for c in checks if c.status == level) for level in ("critical", "warning", "ok")}
    lines.append("=" * 40)
    lines.append(
        f"Summary: {counts['ok']} ok, {counts['warning']} warning(s), {counts['critical']} critical"
    )
    if counts["critical"]:
        lines.append("Resolution: install the missing required packages listed above.")
    elif counts["warning"]:
        lines.append("Tip: optional packages improve features but none are required.")
    else:
        lines.append("All checks passed. The environment is healthy.")
    return "\n".join(lines)


def exit_code_for(checks: list[DoctorCheck]) -> int:
    """Map doctor results to a CI-friendly exit code: 0 healthy, 2 critical."""
    return 2 if any(c.status == "critical" for c in checks) else 0
