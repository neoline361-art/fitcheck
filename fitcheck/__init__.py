"""FitCheck — Zero-boilerplate ML data validation and model evaluation."""

from fitcheck._version import __version__
from fitcheck.backends import get_backend
from fitcheck.check import check
from fitcheck.drift import detect_drift
from fitcheck.extensions import detect_seasonality, run_plugins, validate_timeseries
from fitcheck.fingerprint import fingerprint, hash_file, verify_report
from fitcheck.integrations import log_to_dvc, log_to_mlflow
from fitcheck.plugins import load_plugin, registry
from fitcheck.report import report
from fitcheck.viz import get_renderer

__author__ = "neoline361-art"

__all__ = [
    "check",
    "report",
    "detect_drift",
    "detect_seasonality",
    "run_plugins",
    "validate_timeseries",
    "registry",
    "load_plugin",
    "get_backend",
    "get_renderer",
    "log_to_mlflow",
    "log_to_dvc",
    "verify_report",
    "hash_file",
    "fingerprint",
    "__version__",
]

# Pro features (graceful fallback)
try:
    from fitcheck.fix import generate_fix_script  # noqa: F401

    __all__.append("generate_fix_script")
except ImportError:  # pragma: no cover - fallback for broken installs
    pass
