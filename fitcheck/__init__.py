"""FitCheck — Zero-boilerplate ML data validation and model evaluation."""

from fitcheck.check import check
from fitcheck.drift import detect_drift
from fitcheck.report import report

__version__ = "2.0.0"
__author__ = "neoline361-art"

__all__ = ["check", "report", "detect_drift", "__version__"]

# Pro features (graceful fallback)
try:
    from fitcheck.fix import generate_fix_script  # noqa: F401
    __all__.append("generate_fix_script")
except ImportError:
    pass
