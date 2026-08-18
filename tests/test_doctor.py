"""Tests for the ``fitcheck doctor`` environment diagnosis command."""

from __future__ import annotations


def test_doctor_checks_required_packages() -> None:
    from fitcheck.doctor import run_doctor_checks

    checks = run_doctor_checks()
    names = [c.name for c in checks]
    assert "required:pandas" in names
    assert "required:numpy" in names
    assert any(name.startswith("optional:") for name in names)
    assert any(name.startswith("tool:") for name in names)
    # In this dev environment required packages are present.
    assert all(c.status == "ok" for c in checks if c.name.startswith("required:"))


def test_doctor_format_report_summary() -> None:
    from fitcheck.doctor import format_doctor_report, run_doctor_checks

    report = format_doctor_report(run_doctor_checks())
    assert "FitCheck environment diagnosis" in report
    assert "Summary:" in report
    assert "All checks passed" in report or "warning(s)" in report


def test_doctor_exit_code_healthy() -> None:
    from fitcheck.doctor import exit_code_for, run_doctor_checks

    assert exit_code_for(run_doctor_checks()) == 0


def test_doctor_cli_exits_zero() -> None:
    from fitcheck.cli import main

    assert main(["doctor"]) == 0
    assert main(["doctor", "--json"]) == 0


def test_doctor_detects_missing_package(monkeypatch) -> None:
    """Simulating a missing required package must mark it critical."""
    import importlib

    from fitcheck.doctor import run_doctor_checks

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pandas":
            raise ImportError("simulated missing pandas")
        return original_import(name, *args, **kwargs)

    original_import = importlib.import_module
    monkeypatch.setattr("importlib.import_module", fake_import)
    try:
        checks = run_doctor_checks()
    finally:
        monkeypatch.undo()
    pandas_check = next(c for c in checks if c.name == "required:pandas")
    assert pandas_check.status == "critical"
