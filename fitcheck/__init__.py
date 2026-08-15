"""FitCheck — Zero-boilerplate ML data validation and model evaluation."""

from fitcheck._version import __version__
from fitcheck.check import check
from fitcheck.drift import detect_drift
from fitcheck.extensions import run_plugins, validate_timeseries
from fitcheck.plugins import load_plugin, registry
from fitcheck.report import report

__author__ = "neoline361-art"

__all__ = [
    "check",
    "report",
    "detect_drift",
    "run_plugins",
    "validate_timeseries",
    "registry",
    "load_plugin",
    "__version__",
]

# Pro features (graceful fallback)
try:
    from fitcheck.fix import generate_fix_script  # noqa: F401

    __all__.append("generate_fix_script")
except ImportError:  # pragma: no cover - fallback for broken installs
    pass
