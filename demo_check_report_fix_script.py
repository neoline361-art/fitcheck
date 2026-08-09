"""Auto-generated FitCheck Fix Script

Generated: 2026-08-09 07:55:24
WARNING: Review every step before running.
This script reads from INPUT and writes to OUTPUT.
"""

import os
import pandas as pd
import numpy as np

INPUT_PATH = 'demo_data.csv'
OUTPUT_PATH = 'cleaned_data.csv'


def load_data(path: str) -> pd.DataFrame:
    """Load data, validating existence."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save cleaned data to a NEW file."""
    if os.path.exists(path):
        print(f'WARNING: {path} already exists. Overwriting.')
    if path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f'Saved: {path} ({len(df)} rows, {len(df.columns)} columns)')


def main() -> None:
    print(f'Loading: {INPUT_PATH}')
    df = load_data(INPUT_PATH)
    print(f'Original: {len(df)} rows, {len(df.columns)} columns')
    changes: list[str] = []

    # Step 1: duplicate_rows (info)
    # Column: all
    # Issue: 10 duplicate rows (2.0%)
    # Rationale: Duplicate rows add no information and inflate sample size
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f'  Removed {before - after} duplicate rows')
    changes.append("duplicate_rows on all")

    # Step 2: constant_column (warning)
    # Column: constant_col
    # Issue: constant_col: constant value "42" (zero variance)
    # Rationale: Constant column constant_col has zero variance
    if 'constant_col' in df.columns:
        df = df.drop(columns=['constant_col'])
        print(f'  Dropped constant column: constant_col')
    changes.append("constant_column on constant_col")

    # Summary
    print('')
    print('=== FitCheck Fix Summary ===')
    print(f'Total changes: {len(changes)}')
    for c in changes:
        print(f'  - {c}')
    print(f'')
    print(f'Saving to: {OUTPUT_PATH}')
    save_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
