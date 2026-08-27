"""Tests for fingerprint, file hashing, HMAC signing, and verify CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from fitcheck._version import __version__
from fitcheck.fingerprint import (
    fingerprint,
    fingerprint_html,
    hash_file,
    verify_report,
    _hash_dataframe,
    _hash_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]})


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    path = tmp_path / "test_data.csv"
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_config():
    return {"missing_warning": 0.05, "missing_critical": 0.20}


# ---------------------------------------------------------------------------
# hash_file
# ---------------------------------------------------------------------------

class TestHashFile:
    def test_deterministic(self, sample_csv):
        h1 = hash_file(sample_csv)
        h2 = hash_file(sample_csv)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_files_differ(self, tmp_path, sample_df):
        p1 = tmp_path / "a.csv"
        p2 = tmp_path / "b.csv"
        sample_df.to_csv(p1, index=False)
        sample_df.to_csv(p2, index=False)
        assert hash_file(p1) == hash_file(p2)

        sample_df["d"] = [10, 20, 30]
        sample_df.to_csv(p2, index=False)
        assert hash_file(p1) != hash_file(p2)

    def test_parquet_file_hash(self, tmp_path, sample_df):
        path = tmp_path / "data.parquet"
        sample_df.to_parquet(path, index=False)
        h = hash_file(path)
        assert len(h) == 64
        # Different format from CSV => different hash
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)
        assert h != hash_file(csv_path)


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_basic_fingerprint(self, sample_df, sample_config):
        fp = fingerprint(sample_df, sample_config)
        assert "dataset_hash" in fp
        assert "config_hash" in fp
        assert fp["fitcheck_version"] == __version__
        assert "timestamp" in fp
        assert "signature" not in fp  # no secret_key => no signature

    def test_raw_hash_overrides_dataframe_hash(self, sample_df, sample_config):
        fp_no_raw = fingerprint(sample_df, sample_config)
        fp_with_raw = fingerprint(sample_df, sample_config, raw_hash="abc123")
        assert fp_with_raw["dataset_hash"] == "abc123"
        assert fp_with_raw["dataset_hash"] != fp_no_raw["dataset_hash"]

    def test_hmac_signature_present_when_secret_key(self, sample_df, sample_config):
        fp = fingerprint(sample_df, sample_config, secret_key="my-secret", result_summary="PASS")
        assert "signature" in fp
        assert len(fp["signature"]) == 64  # SHA-256 hex

    def test_hmac_signature_valid_sha256(self, sample_df, sample_config):
        # Both signatures should be valid SHA-256 hex digests
        fp1 = fingerprint(sample_df, sample_config, secret_key="key1", result_summary="PASS")
        fp2 = fingerprint(sample_df, sample_config, secret_key="key1", result_summary="PASS")
        assert len(fp1["signature"]) == 64
        assert len(fp2["signature"]) == 64
        assert all(c in '0123456789abcdef' for c in fp1["signature"])

    def test_different_keys_produce_different_signatures(self, sample_df, sample_config):
        fp1 = fingerprint(sample_df, sample_config, secret_key="key1")
        fp2 = fingerprint(sample_df, sample_config, secret_key="key2")
        assert fp1["signature"] != fp2["signature"]

    def test_result_summary_affects_signature(self, sample_df, sample_config):
        fp1 = fingerprint(sample_df, sample_config, secret_key="key1", result_summary="PASS")
        fp2 = fingerprint(sample_df, sample_config, secret_key="key1", result_summary="3 issues found")
        assert fp1["signature"] != fp2["signature"]


# ---------------------------------------------------------------------------
# fingerprint_html
# ---------------------------------------------------------------------------

class TestFingerprintHtml:
    def test_contains_visible_fingerprint(self, sample_df, sample_config):
        html = fingerprint_html(sample_df, sample_config)
        assert 'class="fingerprint"' in html
        assert "FitCheck" in html
        assert 'class="fc-fingerprint"' in html
        assert "verify: fitcheck verify" in html

    def test_contains_hidden_json(self, sample_df, sample_config):
        html = fingerprint_html(sample_df, sample_config)
        assert 'type="hidden"' in html
        assert 'class="fc-fingerprint"' in html

    def test_hmac_signature_in_html(self, sample_df, sample_config):
        html = fingerprint_html(sample_df, sample_config, secret_key="my-key")
        assert "sig:" in html


# ---------------------------------------------------------------------------
# verify_report
# ---------------------------------------------------------------------------

class TestVerifyReport:
    def test_verify_with_matching_data(self, sample_csv, sample_df, sample_config, tmp_path):
        # Generate report
        from fitcheck.check import check
        report_path = str(tmp_path / "report.html")
        check(str(sample_csv), output=report_path)

        # Verify should pass
        result = verify_report(report_path, str(sample_csv))
        assert result["match"] is True
        assert result["report_version"] == __version__

    def test_verify_with_tampered_data(self, sample_csv, sample_df, sample_config, tmp_path):
        from fitcheck.check import check
        report_path = str(tmp_path / "report.html")
        check(str(sample_csv), output=report_path)

        # Modify the CSV
        tampered = tmp_path / "tampered.csv"
        sample_df["extra"] = [99, 99, 99]
        sample_df.to_csv(tampered, index=False)

        result = verify_report(report_path, str(tampered))
        assert result["match"] is False
        assert "MISMATCH" in result["message"] or "tampered" in result["message"].lower()

    def test_verify_without_source_file(self, sample_csv, sample_df, sample_config, tmp_path):
        from fitcheck.check import check
        report_path = str(tmp_path / "report.html")
        check(str(sample_csv), output=report_path)

        result = verify_report(report_path)
        assert result["match"] is True  # fingerprint exists, no source to compare

    def test_verify_no_fingerprint(self, tmp_path):
        fake_html = tmp_path / "no_fp.html"
        fake_html.write_text("<html><body>No fingerprint here</body></html>")
        result = verify_report(str(fake_html))
        assert result["match"] is False
        assert "No fingerprint" in result["message"]

    def test_verify_hmac_valid(self, sample_csv, sample_df, sample_config, tmp_path):
        from fitcheck.check import check
        report_path = str(tmp_path / "report.html")
        check(str(sample_csv), output=report_path, secret_key="test-secret", return_format="list")

        result = verify_report(report_path, str(sample_csv), secret_key="test-secret")
        assert result["match"] is True
        assert result["signature_valid"] is True

    def test_verify_hmac_invalid_key(self, sample_csv, sample_df, sample_config, tmp_path):
        from fitcheck.check import check
        report_path = str(tmp_path / "report.html")
        check(str(sample_csv), output=report_path, secret_key="correct-key", return_format="list")

        result = verify_report(report_path, str(sample_csv), secret_key="wrong-key")
        assert result["match"] is False
        assert result["signature_valid"] is False
        assert "tampered" in result["message"].lower()

    def test_verify_no_hmac_no_signature_check(self, sample_csv, sample_df, sample_config, tmp_path):
        from fitcheck.check import check
        report_path = str(tmp_path / "report.html")
        check(str(sample_csv), output=report_path)

        result = verify_report(report_path, str(sample_csv))
        assert result["signature_valid"] is None  # no signature to check


# ---------------------------------------------------------------------------
# _hash_dataframe / _hash_dict
# ---------------------------------------------------------------------------

class TestHashHelpers:
    def test_hash_dataframe_deterministic(self, sample_df):
        h1 = _hash_dataframe(sample_df)
        h2 = _hash_dataframe(sample_df)
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_dict_deterministic(self, sample_config):
        h1 = _hash_dict(sample_config)
        h2 = _hash_dict(sample_config)
        assert h1 == h2

    def test_hash_dict_order_independent(self):
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert _hash_dict(d1) == _hash_dict(d2)


class TestDataFrameWarning:
    def test_dataframe_input_warns(self, sample_df, sample_config, tmp_path):
        import warnings
        from fitcheck.check import check
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check(sample_df, output=str(tmp_path / "report.html"))
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert any("DataFrame input" in str(x.message) for x in user_warnings)


# ---------------------------------------------------------------------------
# CLI verify command
# ---------------------------------------------------------------------------

class TestCliVerify:
    def test_cli_verify_valid(self, sample_csv, sample_df, sample_config, tmp_path):
        from fitcheck.check import check
        report_path = str(tmp_path / "report.html")
        check(str(sample_csv), output=report_path)

        result = subprocess.run(
            [sys.executable, "-m", "fitcheck", "verify", report_path, "--against", str(sample_csv)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "VALID" in result.stdout

    def test_cli_verify_tampered(self, sample_csv, sample_df, sample_config, tmp_path):
        from fitcheck.check import check
        report_path = str(tmp_path / "report.html")
        check(str(sample_csv), output=report_path)

        tampered = tmp_path / "tampered.csv"
        sample_df["extra"] = [99, 99, 99]
        sample_df.to_csv(tampered, index=False)

        result = subprocess.run(
            [sys.executable, "-m", "fitcheck", "verify", report_path, "--against", str(tampered)],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "TAMPERED" in result.stdout

    def test_cli_verify_json_output(self, sample_csv, sample_df, sample_config, tmp_path):
        from fitcheck.check import check
        report_path = str(tmp_path / "report.html")
        check(str(sample_csv), output=report_path)

        result = subprocess.run(
            [sys.executable, "-m", "fitcheck", "verify", report_path, "--against", str(sample_csv), "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "match" in data
        assert "report_hash" in data

    def test_cli_check_outputs_verify_hint(self, sample_csv, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "fitcheck", "check", str(sample_csv),
             "--output", str(tmp_path / "out.html"), "--quiet"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        # When not quiet, should print verify hint
        result2 = subprocess.run(
            [sys.executable, "-m", "fitcheck", "check", str(sample_csv),
             "--output", str(tmp_path / "out2.html")],
            capture_output=True, text=True,
        )
        assert "verify" in result2.stdout.lower()

    def test_cli_check_sign_key(self, sample_csv, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "fitcheck", "check", str(sample_csv),
             "--output", str(tmp_path / "signed.html"),
             "--sign-key", "my-secret", "--quiet"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        html_content = (tmp_path / "signed.html").read_text()
        assert "sig:" in html_content
