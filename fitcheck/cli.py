"""Command-line interface for FitCheck.

Exit codes (CI-native):
    0  no issues, or only issues below ``--fail-on``
    1  warnings found (or ``--fail-on warning`` triggered)
    2  critical issues found (or ``--fail-on critical`` triggered)
    3  runtime error (missing file, invalid config, bad arguments)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fitcheck.check import check
from fitcheck.drift import detect_drift
from fitcheck.report import report

_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="fitcheck",
        description="Zero-boilerplate ML data validation and model evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check command
    check_parser = subparsers.add_parser("check", help="Validate dataset quality")
    check_parser.add_argument("data", nargs="+", help="Path(s) to CSV or Parquet file(s)")
    check_parser.add_argument("--target", "-t", default=None, help="Target column name")
    check_parser.add_argument(
        "--output", "-o", default="fitcheck_report.html", help="Output HTML path (single-file checks)"
    )
    check_parser.add_argument("--auto-fix", action="store_true", help="Generate fix script")
    check_parser.add_argument("--missing-warning", type=float, default=0.05)
    check_parser.add_argument("--missing-critical", type=float, default=0.20)
    check_parser.add_argument("--outlier-threshold", type=float, default=0.01)
    check_parser.add_argument("--sample-rows", type=int, default=None, help="Inspect only the first N CSV rows")
    check_parser.add_argument("--time-column", default=None, help="Timestamp column for time-series checks")
    check_parser.add_argument("--plugins", default=None, help="Comma-separated plugin names or dotted module paths")
    check_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout")
    check_parser.add_argument("--quiet", action="store_true", help="Suppress all non-JSON output")
    check_parser.add_argument(
        "--fail-on",
        choices=["info", "warning", "critical"],
        default=None,
        help="Minimum severity that fails the run (default: any issue fails)",
    )

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
    drift_parser.add_argument(
        "--method",
        choices=["auto", "ks", "psi", "wasserstein", "chi2", "js"],
        default="auto",
    )

    # full command
    full_parser = subparsers.add_parser("full", help="Run dataset, model, and optional drift checks")
    full_parser.add_argument("data", help="Dataset path")
    full_parser.add_argument("--target", "-t", required=True, help="Target column name")
    full_parser.add_argument("--model", help="Pickled trained model path (optional)")
    full_parser.add_argument("--reference", help="Optional reference dataset path for drift")
    full_parser.add_argument("--output-dir", default="fitcheck_reports", help="Report directory")
    full_parser.add_argument("--auto-fix", action="store_true", help="Generate fix script")
    full_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # demo command
    subparsers.add_parser("demo", help="Run a quick demo")

    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            return _run_check(args)
        if args.command == "report":
            return _run_report(args)
        if args.command == "drift":
            return _run_drift(args)
        if args.command == "full":
            return _run_full(args)
        if args.command == "demo":
            return _run_demo()
    except (FileNotFoundError, ValueError, KeyError, NameError) as exc:
        print(f"fitcheck: error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # runtime errors surface as exit code 3
        print(f"fitcheck: error: {exc}", file=sys.stderr)
        return 3

    parser.print_help()
    return 1


def _run_check(args: Any) -> int:
    """Execute the check command, returning the CI exit code."""
    files: list[str] = list(args.data)
    if len(files) > 1 and args.output != "fitcheck_report.html":
        raise ValueError("--output requires exactly one data file; omit it for multi-file checks")
    config: dict[str, float] = {
        "missing_warning": args.missing_warning,
        "missing_critical": args.missing_critical,
        "outlier_threshold": args.outlier_threshold,
    }
    plugins = _resolve_plugins(args.plugins)
    return_format = "dict" if args.json else "list"
    max_exit = 0
    results: list[dict[str, Any]] = []

    for path in files:
        output = args.output if len(files) == 1 else f"fitcheck_report_{Path(path).stem}.html"
        result = check(
            data=path,
            target=args.target,
            output=output,
            return_format=return_format,
            auto_fix=args.auto_fix,
            config=config,
            plugins=plugins,
            time_column=args.time_column,
            sample_rows=args.sample_rows,
        )
        if args.json:
            result_dict = cast(dict[str, Any], result)
            results.append(result_dict)
            issues = result_dict.get("issues", [])
        else:
            issues = cast(list[dict[str, Any]], result)
        max_exit = max(max_exit, _exit_code(issues, args.fail_on))
        if not args.json and not args.quiet:
            print(f"Report saved: {output}")
            print(f"Issues found: {len(issues)}")

    if args.json:
        payload: Any = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2, default=str))
    return max_exit


def _run_report(args: Any) -> int:
    """Execute the report command."""
    import pickle  # nosec B403 -- model file is user-supplied, not untrusted input

    with open(args.model, "rb") as f:
        model = pickle.load(f)  # nosec B301 -- loading the user's own model artifact
    x_test_arr = _load_array(args.X_test)
    y_test_arr = _load_array(args.y_test)
    metrics = report(model, x_test_arr, y_test_arr, output=args.output)
    print(f"Model report saved: {args.output}")
    print(f"Metrics: { {k: v for k, v in metrics.items() if k not in ('feature_importance', 'per_class_errors')} }")
    return 0


def _run_drift(args: Any) -> int:
    """Execute the drift command."""
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


def _run_full(args: Any) -> int:
    """Execute the full workflow, writing an executive index report."""
    import pickle  # nosec B403 -- model file is user-supplied

    import pandas as pd

    from fitcheck.html import render_full_html

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data) if not args.data.endswith(".parquet") else pd.read_parquet(args.data)
    dataset = cast(
        dict[str, Any],
        check(
            args.data,  # pass the path so --auto-fix scripts reference the real file
            target=args.target,
            output=str(out_dir / "dataset_report.html"),
            auto_fix=args.auto_fix,
            return_format="dict",
        ),
    )
    summary: dict[str, Any] = {
        "dataset": {
            "issues": len(dataset["issues"]),
            "critical": dataset["summary"]["critical"],
        }
    }
    if args.model:
        with open(args.model, "rb") as file:
            model = pickle.load(file)  # nosec B301 -- user-owned model artifact
        metrics = report(model, data.drop(columns=[args.target]), data[args.target], output=str(out_dir / "model_report.html"))
        summary["model"] = {
            "task": "classification" if "accuracy" in metrics else "regression",
            "metrics": {k: v for k, v in metrics.items() if k not in ("feature_importance", "per_class_errors")},
        }
    else:
        summary["model"] = {"task": "not run"}
    if args.reference:
        drift = detect_drift(args.reference, args.data, output=str(out_dir / "drift_report.html"))
        summary["drift"] = {
            "features": len(drift),
            "drifted": sum(1 for r in drift if r.get("drifted")),
        }
    else:
        summary["drift"] = {"features": 0, "drifted": 0}
    render_full_html(summary, str(out_dir / "index.html"))
    if not args.quiet:
        print(f"Reports saved in: {out_dir}")
        print(f"Executive report: {out_dir / 'index.html'}")
    return 0


def _resolve_plugins(spec: str | None) -> list[Any] | None:
    """Resolve comma-separated plugin names via the plugin loader."""
    if not spec:
        return None
    from fitcheck.plugins import load_plugin

    return [load_plugin(name.strip()) for name in spec.split(",") if name.strip()]


def _exit_code(issues: list[dict[str, Any]], fail_on: str | None) -> int:
    """Map the worst issue severity (and ``--fail-on``) to a CI exit code."""
    worst = max((_SEVERITY_RANK.get(str(issue.get("severity", "info")), 0) for issue in issues), default=0)
    if fail_on is None:
        return worst
    return worst if worst >= _SEVERITY_RANK[fail_on] else 0


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
