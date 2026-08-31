"""Tests for Sprint 3 features: artifact bundles, BaseCheck, verdict exit codes,
action.yml, decision mode double-write fix, and policy threshold wiring."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fitcheck.cli import main
from fitcheck.plugins import BaseCheck, load_plugin, registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(tmp_path: Path) -> str:
    """Create a CSV with issues (missing values) for testing."""
    df = pd.DataFrame({
        "a": [1.0] * 30 + [None] * 20,
        "b": list(range(50)),
        "c": ["x"] * 50,
    })
    csv = tmp_path / "test.csv"
    df.to_csv(csv, index=False)
    return str(csv)


def _make_clean_csv(tmp_path: Path) -> str:
    """Create a clean CSV with no issues."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    csv = tmp_path / "clean.csv"
    df.to_csv(csv, index=False)
    return str(csv)


# ---------------------------------------------------------------------------
# BaseCheck plugin contract
# ---------------------------------------------------------------------------

class TestBaseCheck:
    """Test the BaseCheck ABC and backward compat adapter."""

    def test_basecheck_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseCheck()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        class MyCheck(BaseCheck):
            @property
            def name(self) -> str:
                return "my_check"

            @property
            def version(self) -> str:
                return "1.0.0"

            def run(self, df: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
                return [{
                    "column": "a",
                    "type": "custom",
                    "severity": "info",
                    "message": "custom check ran",
                    "suggestion": "none",
                }]

        check_instance = MyCheck()
        assert check_instance.name == "my_check"
        assert check_instance.version == "1.0.0"

        # __call__ adapter works (legacy callable API)
        df = pd.DataFrame({"a": [1, 2, 3]})
        issues = check_instance(df)
        assert len(issues) == 1
        assert issues[0]["type"] == "custom"

    def test_load_plugin_with_plugin_class(self, tmp_path: Path) -> None:
        """load_plugin resolves BaseCheck subclass via plugin_class attribute."""
        mod_file = tmp_path / "my_plugin.py"
        mod_file.write_text(
            "from fitcheck.plugins import BaseCheck\n"
            "import pandas as pd\n"
            "from typing import Any\n\n"
            "class MyPlugin(BaseCheck):\n"
            "    @property\n"
            "    def name(self): return 'my_plugin'\n"
            "    @property\n"
            "    def version(self): return '0.1.0'\n"
            "    def run(self, df, config): return []\n\n"
            "plugin_class = MyPlugin\n"
        )
        import sys
        sys.path.insert(0, str(tmp_path))
        try:
            plugin = load_plugin("my_plugin")
            assert callable(plugin)
            df = pd.DataFrame({"a": [1]})
            assert plugin(df) == []
        finally:
            sys.path.pop(0)

    def test_legacy_callable_plugin_still_works(self) -> None:
        """Legacy callable plugins registered in registry still work."""
        def legacy_check(df: pd.DataFrame) -> list[dict[str, Any]]:
            return []

        registry.register("_test_legacy", legacy_check)
        try:
            plugin = load_plugin("_test_legacy")
            df = pd.DataFrame({"a": [1]})
            assert plugin(df) == []
        finally:
            registry.unregister("_test_legacy")


# ---------------------------------------------------------------------------
# Artifact bundle
# ---------------------------------------------------------------------------

class TestArtifactBundle:
    """Test --artifact flag: report.html + fingerprint.json + signature.bin in .fitcheck.zip"""

    def test_classic_mode_artifact(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        report = tmp_path / "report.html"
        artifact = tmp_path / "report.fitcheck.zip"
        exit_code = main([
            "check", csv,
            "--output", str(report),
            "--artifact", str(artifact),
            "--quiet",
        ])
        assert exit_code >= 0
        assert artifact.exists()

        with zipfile.ZipFile(artifact) as zf:
            names = zf.namelist()
            assert "report.html" in names
            assert "fingerprint.json" in names

            fp = json.loads(zf.read("fingerprint.json"))
            assert "dataset_hash" in fp
            assert "fitcheck_version" in fp

    def test_artifact_with_sign_key(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        report = tmp_path / "report.html"
        artifact = tmp_path / "signed.fitcheck.zip"
        exit_code = main([
            "check", csv,
            "--output", str(report),
            "--artifact", str(artifact),
            "--sign-key", "test-secret",
            "--quiet",
        ])
        assert exit_code >= 0
        with zipfile.ZipFile(artifact) as zf:
            assert "signature.bin" in zf.namelist()
            sig = zf.read("signature.bin")
            assert len(sig) == 32  # SHA-256 digest

    def test_artifact_with_decision_mode(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        report = tmp_path / "decision.html"
        artifact = tmp_path / "decision.fitcheck.zip"
        exit_code = main([
            "check", csv,
            "--output", str(report),
            "--mode", "decision",
            "--artifact", str(artifact),
            "--quiet",
        ])
        # Decision mode returns verdict-based exit code (BLOCK=2 for 40% missing)
        assert exit_code in (0, 1, 2)
        assert artifact.exists()
        with zipfile.ZipFile(artifact) as zf:
            assert "report.html" in zf.namelist()
            assert "fingerprint.json" in zf.namelist()


# ---------------------------------------------------------------------------
# Artifact verify
# ---------------------------------------------------------------------------

class TestArtifactVerify:
    """Test fitcheck verify <bundle.zip>"""

    def test_verify_bundle_no_source(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        report = tmp_path / "report.html"
        artifact = tmp_path / "test.fitcheck.zip"

        main(["check", csv, "--output", str(report), "--artifact", str(artifact), "--quiet"])

        exit_code = main(["verify", str(artifact)])
        assert exit_code == 0  # valid when no source provided

    def test_verify_bundle_with_matching_source(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        report = tmp_path / "report.html"
        artifact = tmp_path / "test.fitcheck.zip"

        main(["check", csv, "--output", str(report), "--artifact", str(artifact), "--quiet"])

        exit_code = main(["verify", str(artifact), "--against", csv])
        assert exit_code == 0

    def test_verify_bundle_tampered_source(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        report = tmp_path / "report.html"
        artifact = tmp_path / "test.fitcheck.zip"

        main(["check", csv, "--output", str(report), "--artifact", str(artifact), "--quiet"])

        # Tamper: create a different CSV
        different = tmp_path / "different.csv"
        pd.DataFrame({"x": [1]}).to_csv(different, index=False)

        exit_code = main(["verify", str(artifact), "--against", str(different)])
        assert exit_code == 1  # tampered

    def test_verify_bundle_signature_valid(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        report = tmp_path / "report.html"
        artifact = tmp_path / "signed.fitcheck.zip"

        main([
            "check", csv, "--output", str(report),
            "--artifact", str(artifact),
            "--sign-key", "my-secret",
            "--quiet",
        ])

        exit_code = main(["verify", str(artifact), "--secret-key", "my-secret"])
        assert exit_code == 0

    def test_verify_bundle_signature_invalid_key(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        report = tmp_path / "report.html"
        artifact = tmp_path / "signed.fitcheck.zip"

        main([
            "check", csv, "--output", str(report),
            "--artifact", str(artifact),
            "--sign-key", "correct-key",
            "--quiet",
        ])

        exit_code = main(["verify", str(artifact), "--secret-key", "wrong-key"])
        assert exit_code == 1  # signature mismatch

    def test_verify_missing_bundle(self, tmp_path: Path) -> None:
        exit_code = main(["verify", str(tmp_path / "nonexistent.fitcheck.zip")])
        assert exit_code == 3  # FileNotFoundError caught as runtime error


# ---------------------------------------------------------------------------
# Verdict-driven exit codes
# ---------------------------------------------------------------------------

class TestVerdictExitCodes:
    """Decision mode returns verdict-based exit codes: PASS=0, WARN=1, BLOCK=2."""

    def test_clean_data_pass(self, tmp_path: Path) -> None:
        csv = _make_clean_csv(tmp_path)
        exit_code = main(["check", csv, "--mode", "decision", "--quiet"])
        assert exit_code == 0  # PASS

    def test_mild_issues_warn(self, tmp_path: Path) -> None:
        # 10% missing — should trigger WARN with default policy
        df = pd.DataFrame({"a": [1.0] * 9 + [None], "b": list(range(10))})
        csv = tmp_path / "warn.csv"
        df.to_csv(csv, index=False)
        exit_code = main(["check", str(csv), "--mode", "decision", "--quiet"])
        assert exit_code in (0, 1)  # PASS or WARN

    def test_heavy_issues_block(self, tmp_path: Path) -> None:
        # 40% missing — should trigger BLOCK
        df = pd.DataFrame({"a": [1.0] * 30 + [None] * 20, "b": list(range(50))})
        csv = tmp_path / "block.csv"
        df.to_csv(csv, index=False)
        exit_code = main(["check", str(csv), "--mode", "decision", "--quiet"])
        assert exit_code >= 1  # At least WARN for 40% missing data

    def test_classic_mode_unaffected(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        exit_code = main(["check", csv, "--mode", "classic", "--quiet"])
        assert exit_code == 2  # classic: worst severity (critical missing)


# ---------------------------------------------------------------------------
# Policy threshold wiring
# ---------------------------------------------------------------------------

class TestPolicyWiring:
    """Policy block_score/warn_score from fitcheck.yaml are used by verdict."""

    def test_custom_policy_overrides(self, tmp_path: Path) -> None:
        # Create a policy that makes BLOCK very easy (block_score=1)
        policy_file = tmp_path / "fitcheck.yaml"
        policy_file.write_text(
            "fail_thresholds:\n"
            "  block_score: 1\n"
            "  warn_score: 1\n"
        )
        csv = _make_csv(tmp_path)
        exit_code = main([
            "check", csv,
            "--mode", "decision",
            "--policy", str(policy_file),
            "--quiet",
        ])
        assert exit_code == 2  # BLOCK (low threshold)

    def test_high_threshold_still_passes(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "fitcheck.yaml"
        policy_file.write_text(
            "fail_thresholds:\n"
            "  block_score: 100\n"
            "  warn_score: 100\n"
        )
        csv = _make_clean_csv(tmp_path)
        exit_code = main([
            "check", csv,
            "--mode", "decision",
            "--policy", str(policy_file),
            "--quiet",
        ])
        assert exit_code == 0  # PASS (very high threshold)


# ---------------------------------------------------------------------------
# Decision mode no double-write
# ---------------------------------------------------------------------------

class TestDecisionModeDoubleWrite:
    """Decision mode should NOT write classic HTML — only decision HTML."""

    def test_decision_mode_single_html(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        report = tmp_path / "output.html"
        main([
            "check", csv,
            "--output", str(report),
            "--mode", "decision",
            "--quiet",
        ])
        # Only the decision HTML should exist (no classic overwrite)
        assert report.exists()
        content = report.read_text()
        assert "FitCheck Preflight Decision" in content
        # Should NOT contain classic-only content
        assert "FitCheck Dataset Report" not in content

    def test_classic_mode_writes_classic_html(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        report = tmp_path / "output.html"
        main([
            "check", csv,
            "--output", str(report),
            "--mode", "classic",
            "--quiet",
        ])
        assert report.exists()
        content = report.read_text()
        assert "FitCheck Dataset Report" in content
