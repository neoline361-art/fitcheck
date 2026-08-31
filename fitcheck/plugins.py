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
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import pandas as pd

CheckPlugin = Callable[[pd.DataFrame], list[dict[str, Any]]]


class BaseCheck(ABC):
    """Structured, versioned interface for custom checks.

    Subclass this for new checks:

        class MyCheck(BaseCheck):
            @property
            def name(self) -> str:
                return "my_check"

            @property
            def version(self) -> str:
                return "1.0.0"

            def run(self, df: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
                return [{"column": "x", "type": "custom", "severity": "info",
                         "message": "...", "suggestion": "..."}]
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def run(self, df: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]: ...

    def __call__(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Adapter: makes BaseCheck subclasses callable (legacy plugin API)."""
        return self.run(df, {})


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
       ``check``, ``plugin``, ``run``, or ``plugin_class`` attribute
       (in that order) is used. ``plugin_class`` may be a
       :class:`BaseCheck` subclass (instantiated automatically).
    """
    if spec in registry.list():
        return registry.get(spec)
    module = importlib.import_module(spec)
    for attr in ("check", "plugin", "run"):
        candidate = getattr(module, attr, None)
        if callable(candidate):
            return cast(CheckPlugin, candidate)
    # Support BaseCheck subclasses
    cls = getattr(module, "plugin_class", None)
    if cls is not None:
        if isinstance(cls, type) and issubclass(cls, BaseCheck):
            return cast(CheckPlugin, cls())
        if isinstance(cls, BaseCheck):
            return cast(CheckPlugin, cls)
    raise AttributeError(
        f"Module {spec!r} has no callable 'check', 'plugin', 'run', "
        f"or BaseCheck 'plugin_class' attribute"
    )
