"""Transparent, inspectable fix script generation. Never mutates data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FixAction:
    """Immutable fix action descriptor."""

    column: str
    issue_type: str
    severity: str
    description: str
    code: str
    rationale: str


class FixScriptGenerator:
    """Builds transparent Python fix scripts from diagnostic results."""

    def __init__(self, engine: str = "pandas") -> None:
        self.engine = engine
        self.actions: list[FixAction] = []

    def add(self, action: FixAction) -> None:
        """Add a fix action to the script."""
        self.actions.append(action)

    def generate(self, input_path: str, output_path: str = "cleaned_data.csv") -> str:
        """Generate the complete Python fix script as a string."""
        lines: list[str] = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append('"""Auto-generated FitCheck Fix Script')
        lines.append("")
        lines.append(f"Generated: {ts}")
        lines.append("WARNING: Review every step before running.")
        lines.append("This script reads from INPUT and writes to OUTPUT.")
        lines.append('"""')
        lines.append("")
        lines.append("import os")
        lines.append("import pandas as pd")
        lines.append("import numpy as np")
        lines.append("")
        lines.append(f"INPUT_PATH = {repr(input_path)}")
        lines.append(f"OUTPUT_PATH = {repr(output_path)}")
        lines.append("")
        lines.append("")
        lines.append("def load_data(path: str) -> pd.DataFrame:")
        lines.append('    """Load data, validating existence."""')
        lines.append("    if not os.path.exists(path):")
        lines.append('        raise FileNotFoundError(f"File not found: {path}")')
        lines.append("    if path.endswith('.parquet'):")
        lines.append("        return pd.read_parquet(path)")
        lines.append("    return pd.read_csv(path)")
        lines.append("")
        lines.append("")
        lines.append("def save_data(df: pd.DataFrame, path: str) -> None:")
        lines.append('    """Save cleaned data to a NEW file."""')
        lines.append("    if os.path.exists(path):")
        lines.append("        print(f'WARNING: {path} already exists. Overwriting.')")
        lines.append("    if path.endswith('.parquet'):")
        lines.append("        df.to_parquet(path, index=False)")
        lines.append("    else:")
        lines.append("        df.to_csv(path, index=False)")
        lines.append("    print(f'Saved: {path} ({len(df)} rows, {len(df.columns)} columns)')")
        lines.append("")
        lines.append("")
        lines.append("def main() -> None:")
        lines.append("    print(f'Loading: {INPUT_PATH}')")
        lines.append("    df = load_data(INPUT_PATH)")
        lines.append("    print(f'Original: {len(df)} rows, {len(df.columns)} columns')")
        lines.append("    changes: list[str] = []")
        lines.append("")

        for i, action in enumerate(self.actions, 1):
            lines.append(f"    # Step {i}: {action.issue_type} ({action.severity})")
            lines.append(f"    # Column: {action.column}")
            lines.append(f"    # Issue: {action.description}")
            lines.append(f"    # Rationale: {action.rationale}")
            for code_line in action.code.split("\n"):
                lines.append(f"    {code_line}")
            lines.append(f'    changes.append("{action.issue_type} on {action.column}")')
            lines.append("")

        lines.append("    # Summary")
        lines.append("    print('')")
        lines.append("    print('=== FitCheck Fix Summary ===')")
        lines.append("    print(f'Total changes: {len(changes)}')")
        lines.append("    for c in changes:")
        lines.append("        print(f'  - {c}')")
        lines.append("    print(f'')")
        lines.append("    print(f'Saving to: {OUTPUT_PATH}')")
        lines.append("    save_data(df, OUTPUT_PATH)")
        lines.append("")
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    main()")
        lines.append("")

        return "\n".join(lines)

    def save(
        self, input_path: str, script_path: str, output_path: str = "cleaned_data.csv"
    ) -> Path:
        """Generate and write the fix script to disk."""
        script = self.generate(input_path, output_path)
        path = Path(script_path)
        path.write_text(script, encoding="utf-8")
        return path


def generate_fix_script(
    diagnostics: dict[str, Any], input_path: str, script_path: str = "fitcheck_fix_script.py"
) -> str:
    """Convert diagnostics into a fix script."""
    generator = FixScriptGenerator()
    for issue in diagnostics.get("issues", []):
        action = _to_action(issue)
        if action:
            generator.add(action)

    if not generator.actions:
        generator.add(
            FixAction(
                column="none",
                issue_type="no_action_needed",
                severity="info",
                description="No actionable issues detected",
                code="pass  # No fixes required",
                rationale="Dataset passed all checks",
            )
        )

    generator.save(input_path, script_path)
    return generator.generate(input_path)


def _to_action(issue: dict[str, Any]) -> FixAction | None:
    """Convert a diagnostic issue dict into a FixAction."""
    itype = issue.get("type", "")
    col = issue.get("column", "")
    sev = issue.get("severity", "info")
    msg = issue.get("message", "")

    if itype == "missing_values":
        return FixAction(
            column=col,
            issue_type="missing_values",
            severity=sev,
            description=msg,
            code=(
                f"if '{col}' in df.columns:\n"
                f"    median_val = df['{col}'].median()\n"
                f"    df['{col}'] = df['{col}'].fillna(median_val)\n"
                f"    print(f'  Filled missing in {col} with median: {{median_val}}')"
            ),
            rationale=f"Median imputation preserves distribution for {col}",
        )
    elif itype == "duplicate_rows":
        return FixAction(
            column="all",
            issue_type="duplicate_rows",
            severity=sev,
            description=msg,
            code=(
                "before = len(df)\n"
                "df = df.drop_duplicates()\n"
                "after = len(df)\n"
                "print(f'  Removed {before - after} duplicate rows')"
            ),
            rationale="Duplicate rows add no information and inflate sample size",
        )
    elif itype == "constant_column":
        return FixAction(
            column=col,
            issue_type="constant_column",
            severity=sev,
            description=msg,
            code=(
                f"if '{col}' in df.columns:\n"
                f"    df = df.drop(columns=['{col}'])\n"
                f"    print(f'  Dropped constant column: {col}')"
            ),
            rationale=f"Constant column {col} has zero variance",
        )
    elif itype == "outliers":
        return FixAction(
            column=col,
            issue_type="outliers",
            severity=sev,
            description=msg,
            code=(
                f"if '{col}' in df.columns:\n"
                f"    q1 = df['{col}'].quantile(0.25)\n"
                f"    q3 = df['{col}'].quantile(0.75)\n"
                f"    iqr = q3 - q1\n"
                f"    lower = q1 - 1.5 * iqr\n"
                f"    upper = q3 + 1.5 * iqr\n"
                f"    df['{col}'] = df['{col}'].clip(lower, upper)\n"
                f"    print(f'  Capped outliers in {col}')"
            ),
            rationale=f"IQR capping reduces extreme values in {col}",
        )
    elif itype == "class_imbalance":
        return FixAction(
            column=col,
            issue_type="class_imbalance",
            severity=sev,
            description=msg,
            code=f"# Class imbalance in {col}: use SMOTE or class_weight",
            rationale="Address imbalance during model training",
        )
    return None
