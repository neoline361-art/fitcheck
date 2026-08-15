"""IPython magics: ``%fitcheck`` and ``%%fitcheck``.

The extension loads lazily so the core package never depends on IPython::

    %load_ext fitcheck
    %fitcheck df --target label
    %%fitcheck --target label

The cell magic runs the full check on the ``df`` variable in the notebook
namespace and renders the report inline.
"""

from __future__ import annotations

import importlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from fitcheck.check import check


def _ipython_module(name: str) -> Any:
    """Import an IPython submodule lazily.

    IPython ships no type stubs, so the module is exposed as ``Any`` to keep
    mypy strict happy whether IPython is installed (with or without stubs) or
    absent entirely.
    """
    return importlib.import_module(name)


def _user_ns() -> dict[str, Any]:
    """Return the active notebook namespace, falling back to globals()."""
    ipython: Any = None
    try:
        ipython = _ipython_module("IPython").get_ipython()
    except Exception:  # pragma: no cover - defensive, IPython unavailable
        ipython = None
    if ipython is not None:
        ns = ipython.user_ns
        if isinstance(ns, dict):
            return ns
    return globals()


def _parse_args(line: str) -> tuple[str | None, str | None]:
    """Parse ``<variable> [--target NAME]``; the variable is the first token."""
    tokens = line.split()
    target: str | None = None
    variable: str | None = None
    if tokens and not tokens[0].startswith("--"):
        variable = tokens[0]
    for i, token in enumerate(tokens):
        if token == "--target" and i + 1 < len(tokens):  # nosec B105 -- flag comparison, not a credential
            target = tokens[i + 1]
    return variable, target


def _resolve_data(variable: str | None, ns: dict[str, Any]) -> pd.DataFrame:
    """Resolve the DataFrame to check from the namespace."""
    candidates = [variable] if variable else [name for name in ("df", "data") if name in ns]
    for name in candidates:
        value = ns.get(name)
        if isinstance(value, pd.DataFrame):
            return value
    raise NameError(
        f"Could not find a DataFrame named {candidates[0] if candidates else 'df'} in the notebook namespace"
    )


def _render_inline(df: pd.DataFrame, target: str | None, display_fn: Callable[[Any], Any]) -> None:
    """Run the check and display the resulting HTML in the notebook."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = Path(tmp_dir) / "fitcheck_report.html"
        check(df, target=target, output=str(report_path))
        html = _ipython_module("IPython.display").HTML(report_path.read_text(encoding="utf-8"))
        display_fn(html)


def run_fitcheck_line(line: str, ns: dict[str, Any] | None = None) -> None:
    """Line magic: ``%fitcheck df --target label``."""
    namespace = ns if ns is not None else _user_ns()
    variable, target = _parse_args(line)
    df = _resolve_data(variable, namespace)
    _render_inline(df, target, _display_or_print)


def run_fitcheck_cell(line: str, cell: str, ns: dict[str, Any] | None = None) -> None:
    """Cell magic: ``%%fitcheck --target label`` (uses the ``df`` variable)."""
    namespace = ns if ns is not None else _user_ns()
    _, target = _parse_args(line)
    df = _resolve_data(None, namespace)
    _render_inline(df, target, _display_or_print)


def _display_or_print(html: Any) -> None:
    """Display inline when IPython is present, otherwise print a note."""
    try:
        _ipython_module("IPython.display").display(html)
    except Exception:  # pragma: no cover - plain Python session
        print("FitCheck report generated; install IPython to render inline.")


def load_ipython_extension(ipython: Any) -> None:
    """IPython extension entry point: register both magics."""
    ipython.register_magic_function(
        run_fitcheck_line, magic_kind="line", magic_name="fitcheck"
    )
    ipython.register_magic_function(
        run_fitcheck_cell, magic_kind="cell", magic_name="fitcheck"
    )
