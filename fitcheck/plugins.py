"""Lightweight plugin registry for custom FitCheck checks.

A plugin is any callable taking a pandas DataFrame and returning a list of
issue dictionaries (matching the check engine's issue schema).

Usage::

    from fitcheck.plugins import registry, load_plugin

    registry.register("domain", my_domain_check)
    check("data.csv", plugins=[load_plugin("domain")])
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

import pandas as pd

CheckPlugin = Callable[[pd.DataFrame], list[dict[str, Any]]]


class PluginRegistry:
    """Named registry of custom checks, registerable at runtime."""

    def __init__(self) -> None:
        self._checks: dict[str, CheckPlugin] = {}

    def register(self, name: str, check: CheckPlugin) -> None:
        """Register a check under ``name``."""
        if not callable(check):
            raise TypeError(f"Plugin {name!r} must be callable")
        self._checks[name] = check

    def get(self, name: str) -> CheckPlugin:
        """Return the registered check or raise KeyError."""
        if name not in self._checks:
            raise KeyError(f"No plugin registered under {name!r}")
        return self._checks[name]

    def list(self) -> list[str]:
        """Return registered plugin names, sorted."""
        return sorted(self._checks)

    def unregister(self, name: str) -> None:
        """Remove a registered check."""
        del self._checks[name]


registry = PluginRegistry()


def load_plugin(spec: str) -> CheckPlugin:
    """Resolve ``spec`` to a check callable.

    Resolution order:
    1. A name registered in the global :data:`registry`.
    2. A dotted module path such as ``my_pkg.my_checks``; the module's
       ``check``, ``plugin``, or ``run`` attribute (in that order) is used.
    """
    if spec in registry.list():
        return registry.get(spec)
    module = importlib.import_module(spec)
    for attr in ("check", "plugin", "run"):
        candidate = getattr(module, attr, None)
        if callable(candidate):
            return cast(CheckPlugin, candidate)
    raise AttributeError(
        f"Module {spec!r} has no callable 'check', 'plugin', or 'run' attribute"
    )
