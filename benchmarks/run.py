"""Reproducible benchmarks for FitCheck.

Usage:
    python benchmarks/run.py

Output:
    benchmarks/results.md (appended)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fitcheck

BENCHMARKS_DIR = Path(__file__).parent
RESULTS_FILE = BENCHMARKS_DIR / "results.md"


def _make_dataset(n_rows: int, n_cols: int, missing_pct: float = 0.05) -> pd.DataFrame:
    """Create a synthetic dataset with configurable size and missing rate."""
    np.random.seed(42)
    data = np.random.randn(n_rows, n_cols)
    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_cols)])
    df["label"] = np.random.randint(0, 2, n_rows)
    # Inject missing values
    mask = np.random.random((n_rows, n_cols)) < missing_pct
    df.iloc[:, :n_cols] = df.iloc[:, :n_cols].mask(mask)
    return df


def benchmark_check() -> dict:
    """Benchmark dataset health check across sizes."""
    results = []
    for rows, cols in [(1000, 10), (10000, 20), (100000, 30)]:
        df = _make_dataset(rows, cols)
        start = time.perf_counter()
        fitcheck.check(df, target="label", output="/dev/null")
        elapsed = time.perf_counter() - start
        results.append({
            "dataset": f"{rows} rows x {cols} cols",
            "time_s": round(elapsed, 3),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        })
    return {"check": results}


def benchmark_drift() -> dict:
    """Benchmark drift detection across sizes."""
    results = []
    for rows, cols in [(1000, 10), (10000, 20), (100000, 30)]:
        ref = _make_dataset(rows, cols)
        prod = _make_dataset(rows, cols)
        start = time.perf_counter()
        fitcheck.detect_drift(ref, prod, output="/dev/null")
        elapsed = time.perf_counter() - start
        results.append({
            "dataset": f"{rows} rows x {cols} cols",
            "time_s": round(elapsed, 3),
        })
    return {"drift": results}


def main():
    print("Running FitCheck benchmarks...")
    results = {}
    results.update(benchmark_check())
    results.update(benchmark_drift())
    print(json.dumps(results, indent=2))

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if RESULTS_FILE.exists() else "w"
    with open(RESULTS_FILE, mode) as f:
        f.write(f"## Benchmark Run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for category, benches in results.items():
            f.write(f"### {category}\n\n")
            f.write("| Dataset | Time (s) | Memory (MB) |\n")
            f.write("|---------|----------|-------------|\n")
            for b in benches:
                mem = b.get("memory_mb", "—")
                f.write(f"| {b['dataset']} | {b['time_s']} | {mem} |\n")
            f.write("\n")
        f.write("---\n\n")

    print(f"Results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
