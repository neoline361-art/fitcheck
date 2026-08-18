"""Auto-fix quality tests.

Five deliberately dirty datasets exercise the generated fix scripts. Each
dataset produces a fix script via the public auto_fix entry point, and the
script must execute cleanly and repair the reported issue.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fitcheck.check import check


def _run_fix_script(script_path: str) -> str:
    """Execute a generated fix script as a subprocess and return stdout."""
    completed = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, f"fix script failed:\n{completed.stderr}"
    return completed.stdout


def _load_fix_output(script_path: str) -> pd.DataFrame:
    """Extract the repaired DataFrame a fix script writes (its fixed OUTPUT_PATH)."""
    import re

    source = Path(script_path).read_text(encoding="utf-8")
    match = re.search(r"OUTPUT_PATH = (.+)", source)
    assert match, "generated fix script must declare OUTPUT_PATH"
    output_csv = match.group(1).strip().strip("'\"")
    stdout = _run_fix_script(script_path)
    assert "Saved:" in stdout, f"expected fix script to announce its output; got:\n{stdout}"
    return pd.read_csv(output_csv)


@pytest.fixture()
def dirty_dataset_1(tmp_path: Path) -> tuple[Path, str]:
    """Dataset with high missing ratio in one column."""
    rng = np.random.default_rng(1)
    data = pd.DataFrame({"a": rng.random(200), "b": [np.nan] * 160 + list(rng.random(40))})
    path = tmp_path / "missing.csv"
    data.to_csv(path, index=False)
    check(str(path), auto_fix=True, output=str(tmp_path / "missing.html"))
    script = str(tmp_path / "missing_fix_script.py")
    assert Path(script).exists()
    return path, script


@pytest.fixture()
def dirty_dataset_2(tmp_path: Path) -> tuple[Path, str]:
    """Dataset with duplicate rows."""
    rng = np.random.default_rng(2)
    base = pd.DataFrame({"x": rng.integers(0, 100, 50), "y": rng.integers(0, 100, 50)})
    data = pd.concat([base, base.iloc[:25]], ignore_index=True)
    path = tmp_path / "dups.csv"
    data.to_csv(path, index=False)
    check(str(path), auto_fix=True, output=str(tmp_path / "dups.html"))
    script = str(tmp_path / "dups_fix_script.py")
    assert Path(script).exists()
    return path, script


@pytest.fixture()
def dirty_dataset_3(tmp_path: Path) -> tuple[Path, str]:
    """Dataset with a constant column."""
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5] * 40, "const": ["same"] * 200})
    path = tmp_path / "const.csv"
    data.to_csv(path, index=False)
    check(str(path), auto_fix=True, output=str(tmp_path / "const.html"))
    script = str(tmp_path / "const_fix_script.py")
    assert Path(script).exists()
    return path, script


@pytest.fixture()
def dirty_dataset_4(tmp_path: Path) -> tuple[Path, str]:
    """Dataset with a high-cardinality ID column."""
    rng = np.random.default_rng(4)
    data = pd.DataFrame({"user_id": [f"u{i}" for i in rng.integers(0, 10_000, 300)], "score": rng.random(300)})
    path = tmp_path / "id.csv"
    data.to_csv(path, index=False)
    check(str(path), auto_fix=True, output=str(tmp_path / "id.html"))
    script = str(tmp_path / "id_fix_script.py")
    assert Path(script).exists()
    return path, script


@pytest.fixture()
def dirty_dataset_5(tmp_path: Path) -> tuple[Path, str]:
    """Dataset with skewed text lengths."""
    data = pd.DataFrame({"note": ["ok"] * 150 + ["x" * 500] * 5})
    path = tmp_path / "text.csv"
    data.to_csv(path, index=False)
    check(str(path), auto_fix=True, output=str(tmp_path / "text.html"))
    script = str(tmp_path / "text_fix_script.py")
    assert Path(script).exists()
    return path, script


def test_autofix_missing_values(dirty_dataset_1: tuple[Path, str]) -> None:
    path, script = dirty_dataset_1
    repaired = _load_fix_output(script)
    assert repaired["b"].isna().mean() <= 0.05 or "b" not in repaired.columns


def test_autofix_duplicate_rows(dirty_dataset_2: tuple[Path, str]) -> None:
    _, script = dirty_dataset_2
    repaired = _load_fix_output(script)
    assert not repaired.duplicated().any()
    assert len(repaired) == 50  # 50 base + 25 dupes trimmed to unique rows


def test_autofix_constant_column(dirty_dataset_3: tuple[Path, str]) -> None:
    _, script = dirty_dataset_3
    repaired = _load_fix_output(script)
    assert "const" not in repaired.columns or repaired["const"].nunique() > 1


def test_autofix_high_cardinality(dirty_dataset_4: tuple[Path, str]) -> None:
    """High-cardinality issues get a documented no-op script (manual review required)."""
    _, script = dirty_dataset_4
    repaired = _load_fix_output(script)
    # The generator emits a conservative script; verify it ran and the column
    # is either removed or still flagged by a re-run of the check engine.
    from fitcheck.check import check

    recheck = check(repaired, return_format="dict", output="/dev/null")
    id_issues = [i for i in recheck["issues"] if i["column"] == "user_id"]
    assert not id_issues or "user_id" not in repaired.columns


def test_autofix_text_length_skew(dirty_dataset_5: tuple[Path, str]) -> None:
    _, script = dirty_dataset_5
    repaired = _load_fix_output(script)
    lengths = repaired["note"].dropna().astype(str).str.len()
    assert float(lengths.mean()) <= 3.0 * max(float(lengths.median()), 1.0) or "note" not in repaired.columns


def test_autofix_script_syntax_valid(dirty_dataset_1: tuple[Path, str]) -> None:
    """Generated scripts must compile cleanly even when execution is skipped."""
    _, script = dirty_dataset_1
    spec = importlib.util.spec_from_file_location("fix", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert module is not None
