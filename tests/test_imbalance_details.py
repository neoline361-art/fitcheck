"""Targeted mutation-coverage tests for _detect_imbalance.

These tests assert on the issue payload's shape and message text so that
string- and key-mutants in ``fitcheck/check.py::_detect_imbalance`` cannot
survive.
"""
from pathlib import Path

import pandas as pd

from fitcheck.check import check


def test_imbalance_issue_payload_shape(tmp_path: Path) -> None:
    df = pd.DataFrame({"feat": [1, 2, 3, 4, 5], "label": [0, 0, 0, 0, 1]})
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    result = check(str(csv), target="label", output=str(tmp_path / "report.html"))
    issues = [i for i in result if i.get("type") == "class_imbalance"]
    assert issues
    issue = issues[0]
    # Payload keys must exist (kills dict-key mutations).
    assert issue.get("column") == "label"
    assert "message" in issue and "majority class" in str(issue["message"])
    assert issue.get("severity") == "warning"
    assert "suggestion" in issue and "SMOTE" in str(issue["suggestion"])
    assert "message" in issue and "80" in str(issue["message"])


def test_no_imbalance_on_balanced_target(tmp_path: Path) -> None:
    df = pd.DataFrame({"feat": [1, 2, 3, 4], "label": [0, 0, 1, 1]})
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    result = check(str(csv), target="label", output=str(tmp_path / "report.html"))
    assert not any(i.get("type") == "class_imbalance" for i in result)
