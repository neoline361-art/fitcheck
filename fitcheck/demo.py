"""FitCheck demo — generates all three report types in one run.

1. Dataset health check (with intentional issues)
2. Model evaluation (classification)
3. Drift detection

Usage:
    fitcheck demo [--no-browser] [--output-dir DIR]
    python -m fitcheck.demo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import fitcheck


def run_demo(no_browser: bool = False, output_dir: str | None = None) -> None:
    """Generate the three demo reports, optionally opening them in a browser."""
    out_dir = Path(output_dir) if output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FitCheck Demo")
    print("=" * 60)

    # 1. Dataset with intentional issues
    print("\n[1/3] Creating synthetic dataset with issues...")
    np.random.seed(42)
    n = 500
    df = pd.DataFrame(
        {
            "age": np.concatenate([np.random.normal(35, 10, n - 20), [np.nan] * 20]),
            "income": np.random.normal(50000, 15000, n),
            "score": np.random.normal(700, 100, n),
            "constant_col": [42] * n,  # constant column
            "label": [0] * (n // 4 * 3) + [1] * (n // 4),  # 75/25 imbalance
        }
    )
    # Add duplicates
    df = pd.concat([df, df.head(10)], ignore_index=True)
    data_csv = out_dir / "demo_data.csv"
    df.to_csv(data_csv, index=False)
    print(f"    Saved: {data_csv} ({len(df)} rows, {len(df.columns)} columns)")

    # Run check (same [1/3] section, no duplicate counter)
    print("    Running dataset health check...")
    check_report = out_dir / "demo_check_report.html"
    issues = fitcheck.check(
        str(data_csv), target="label", output=str(check_report), auto_fix=True
    )
    print(f"    Issues found: {len(issues)}")
    print(f"    Report: {check_report}")
    fix_script = out_dir / "demo_check_report_fix_script.py"
    if fix_script.exists():
        print(f"    Fix script: {fix_script}")

    # 2. Model evaluation
    print("\n[2/3] Training model and evaluating...")
    clean_df = df.dropna()
    x = clean_df.drop(columns=["label", "constant_col"])
    y = clean_df["label"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(x_train, y_train)
    model_report = out_dir / "demo_model_report.html"
    metrics = fitcheck.report(model, x_test, y_test, output=str(model_report))
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1 Score: {metrics['f1']:.4f}")
    print(f"    Report: {model_report}")

    # 3. Drift detection
    print("\n[3/3] Detecting distribution drift...")
    ref = pd.DataFrame({"feat_a": np.random.normal(0, 1, 300), "feat_b": np.random.normal(5, 2, 300)})
    prod = pd.DataFrame(
        {"feat_a": np.random.normal(0, 1, 300), "feat_b": np.random.normal(8, 2, 300)}
    )  # shifted mean
    drift_report = out_dir / "demo_drift_report.html"
    results = fitcheck.detect_drift(ref, prod, output=str(drift_report))
    drifted = sum(1 for r in results if r["drifted"])
    print(f"    Features tested: {len(results)}, Drifted: {drifted}")
    print(f"    Report: {drift_report}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    if not no_browser:
        import webbrowser

        index = (out_dir / "demo_check_report.html").resolve()
        if index.exists():
            webbrowser.open(index.as_uri())


def main(argv: list[str] | None = None) -> int:
    """Minimal standalone entry point (``python -m fitcheck.demo``)."""
    import argparse

    parser = argparse.ArgumentParser(prog="fitcheck demo")
    parser.add_argument("--no-browser", action="store_true", help="Skip opening the browser")
    parser.add_argument("--output-dir", default=None, help="Directory for generated reports")
    args = parser.parse_args(argv)
    run_demo(no_browser=args.no_browser, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
