"""Command-line interface for FitCheck."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fitcheck.check import check
from fitcheck.drift import detect_drift
from fitcheck.report import report


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="fitcheck",
        description="Zero-boilerplate ML data validation and model evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check command
    check_parser = subparsers.add_parser("check", help="Validate dataset quality")
    check_parser.add_argument("data", help="Path to CSV or Parquet file")
    check_parser.add_argument("--target", "-t", default=None, help="Target column name")
    check_parser.add_argument(
        "--output", "-o", default="fitcheck_report.html", help="Output HTML path"
    )
    check_parser.add_argument("--auto-fix", action="store_true", help="Generate fix script")
    check_parser.add_argument("--missing-warning", type=float, default=0.05)
    check_parser.add_argument("--missing-critical", type=float, default=0.20)
    check_parser.add_argument("--outlier-threshold", type=float, default=0.01)

    # report command
    report_parser = subparsers.add_parser("report", help="Evaluate a trained model")
    report_parser.add_argument("model", help="Path to pickled model file")
    report_parser.add_argument("X_test", help="Path to X_test (.npy or .csv)")
    report_parser.add_argument("y_test", help="Path to y_test (.npy or .csv)")
    report_parser.add_argument(
        "--output", "-o", default="model_report.html", help="Output HTML path"
    )

    # drift command
    drift_parser = subparsers.add_parser("drift", help="Detect distribution drift")
    drift_parser.add_argument("reference", help="Reference dataset path")
    drift_parser.add_argument("production", help="Production dataset path")
    drift_parser.add_argument(
        "--output", "-o", default="drift_report.html", help="Output HTML path"
    )
    drift_parser.add_argument("--threshold", type=float, default=0.05, help="P-value threshold")
    drift_parser.add_argument("--method", choices=["auto", "ks", "psi", "wasserstein", "chi2"], default="auto")

    # full command
    full_parser = subparsers.add_parser("full", help="Run dataset, model, and optional drift checks")
    full_parser.add_argument("data", help="Dataset path")
    full_parser.add_argument("--target", "-t", required=True, help="Target column name")
    full_parser.add_argument("--model", required=True, help="Pickled trained model path")
    full_parser.add_argument("--reference", help="Optional reference dataset path for drift")
    full_parser.add_argument("--output-dir", default="fitcheck_reports", help="Report directory")
    full_parser.add_argument("--auto-fix", action="store_true", help="Generate fix script")

    # demo command
    subparsers.add_parser("demo", help="Run a quick demo")

    args = parser.parse_args(argv)

    if args.command == "check":
        result = check(
            data=args.data,
            target=args.target,
            output=args.output,
            auto_fix=args.auto_fix,
            config={
                "missing_warning": args.missing_warning,
                "missing_critical": args.missing_critical,
                "outlier_threshold": args.outlier_threshold,
            },
        )
        print(f"Report saved: {args.output}")
        if result:
            print(f"Issues found: {len(result) if hasattr(result, '__len__') else 'see report'}")
        return 0

    elif args.command == "report":
        import pickle  # nosec B403 -- model file is user-supplied, not untrusted input

        with open(args.model, "rb") as f:
            model = pickle.load(f)  # nosec B301 -- loading the user's own model artifact
        x_test_arr = _load_array(args.X_test)
        y_test_arr = _load_array(args.y_test)
        metrics = report(model, x_test_arr, y_test_arr, output=args.output)
        print(f"Model report saved: {args.output}")
        print(f"Metrics: { {k: v for k, v in metrics.items() if k != 'feature_importance'} }")
        return 0

    elif args.command == "drift":
        results = detect_drift(
            reference=args.reference,
            production=args.production,
            output=args.output,
            threshold=args.threshold,
            method=args.method,
        )
        drifted = sum(1 for r in results if r.get("drifted"))
        print(f"Drift report saved: {args.output}")
        print(f"Features tested: {len(results)}, Drifted: {drifted}")
        return 0

    elif args.command == "full":
        import pickle  # nosec B403 -- model file is user-supplied
        from pathlib import Path

        import pandas as pd
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        data = pd.read_csv(args.data) if not args.data.endswith(".parquet") else pd.read_parquet(args.data)
        check(data, target=args.target, output=str(out_dir / "dataset_report.html"), auto_fix=args.auto_fix)
        with open(args.model, "rb") as file:
            model = pickle.load(file)  # nosec B301 -- user-owned model artifact
        metrics = report(model, data.drop(columns=[args.target]), data[args.target], output=str(out_dir / "model_report.html"))
        if args.reference:
            detect_drift(args.reference, args.data, output=str(out_dir / "drift_report.html"))
        print(f"Reports saved in: {out_dir}")
        print(f"Model metrics: { {k: v for k, v in metrics.items() if k != 'feature_importance'} }")
        return 0

    elif args.command == "demo":
        return _run_demo()

    else:
        parser.print_help()
        return 1


def _load_array(path: str) -> NDArray[Any]:
    """Load numpy array from .npy or .csv file."""
    if path.endswith(".npy"):
        return np.asarray(np.load(path))
    import pandas as pd

    df = pd.read_csv(path)
    return np.asarray(df.values)


def _run_demo() -> int:
    """Run the built-in demo."""
    demo_path = Path(__file__).parent / "demo.py"
    if demo_path.exists():
        import runpy

        runpy.run_path(str(demo_path), run_name="__main__")
        return 0
    print("Demo script not found.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
