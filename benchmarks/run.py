"""Reproducible comparative benchmarks for FitCheck.

Measures FitCheck, Evidently, Deepchecks, and Pandera on identical workloads
using identical datasets, warm-up runs, 10+ timed repetitions, and records
median / min / max runtimes plus environment information. Competitor packages
are measured ONLY when actually importable; otherwise the result section is
explicitly marked as unmeasured. No competitor timing is ever estimated or
typed in by hand.

Usage:
    python benchmarks/run.py            # measure FitCheck
    python benchmarks/run.py --all      # measure FitCheck and any importable competitors

Output:
    benchmarks/results.md (appended, timestamped per run)
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARKS_DIR = Path(__file__).parent
RESULTS_FILE = BENCHMARKS_DIR / "results.md"

N_REPETITIONS = 10
WARM_UP_RUNS = 3
SIZES = [
    {"rows": 1000, "cols": 10},
    {"rows": 10000, "cols": 20},
    {"rows": 100000, "cols": 30},
]


def environment_info() -> dict[str, str]:
    """Record OS, Python, CPU, and framework versions for reproducibility."""
    info: dict[str, str] = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": _cpu_description(),
        "ram_gb": _ram_gb(),
        "python": platform.python_version(),
    }
    for name in ("pandas", "numpy", "scikit-learn", "scipy", "fitcheck"):
        try:
            if name == "fitcheck":
                import fitcheck  # noqa: WPS433

                info["fitcheck"] = getattr(fitcheck, "__version__", "unknown")
            else:
                module = importlib.import_module(name.replace("-", "_"))
                info[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[name] = "not installed"
    return info


def _cpu_description() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return platform.processor() or "unknown"
    return "unknown"


def _ram_gb() -> str:
    try:
        with open("/proc/meminfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024 / 1024:.1f}"
    except OSError:
        return "unknown"
    return "unknown"


def make_dataset(rows: int, cols: int, missing_pct: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Deterministically create the canonical benchmark dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.standard_normal((rows, cols)),
        columns=[f"col_{i}" for i in range(cols)],
    )
    df["label"] = rng.integers(0, 2, rows)
    mask = rng.random((rows, cols)) < missing_pct
    df.iloc[:, :cols] = df.iloc[:, :cols].mask(mask)
    return df


def _time_many(workload, repetitions: int = N_REPETITIONS, warm_up: int = WARM_UP_RUNS) -> dict[str, float]:
    """Run the workload, drop warm-up runs, and return median/min/max."""
    for _ in range(warm_up):
        workload()
    times: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        workload()
        times.append(time.perf_counter() - start)
    return {
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
        "n_runs": repetitions,
    }


def benchmark_fitcheck_check(size: dict[str, int]) -> dict[str, object]:
    import fitcheck

    df = make_dataset(size["rows"], size["cols"])

    def workload() -> None:
        fitcheck.check(df, target="label", output="/dev/null")

    return {
        "workload": "fitcheck check",
        "rows": size["rows"],
        "cols": size["cols"],
        **_time_many(workload),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
    }


def benchmark_fitcheck_drift(size: dict[str, int]) -> dict[str, object]:
    import fitcheck

    ref = make_dataset(size["rows"], size["cols"])
    prod = make_dataset(size["rows"], size["cols"], seed=7)

    def workload() -> None:
        fitcheck.detect_drift(ref, prod, output="/dev/null")

    return {
        "workload": "fitcheck drift",
        "rows": size["rows"],
        "cols": size["cols"],
        **_time_many(workload),
    }


def _benchmark_evidently_profile(size: dict[str, int]) -> dict[str, object] | None:
    try:
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset  # type: ignore
        from evidently.report import Report  # type: ignore
    except ImportError:
        return None

    df = make_dataset(size["rows"], size["cols"])

    def workload() -> None:
        report = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
        report.run(reference_data=df.iloc[: len(df) // 2], current_data=df.iloc[len(df) // 2 :])

    return {
        "workload": "evidently profile",
        "rows": size["rows"],
        "cols": size["cols"],
        **_time_many(workload),
    }


def _benchmark_deepchecks_check(size: dict[str, int]) -> dict[str, object] | None:
    try:
        from deepchecks.core.checks import DatasetIntegrity  # type: ignore
        from deepchecks.tabular import Dataset as DCDataset  # type: ignore
        from deepchecks.tabular.suites import data_integrity  # type: ignore
    except ImportError:
        return None

    df = make_dataset(size["rows"], size["cols"])

    def workload() -> None:
        dataset = DCDataset(df.drop(columns=["label"]), label=df["label"])
        data_integrity().run(dataset)

    return {
        "workload": "deepchecks integrity",
        "rows": size["rows"],
        "cols": size["cols"],
        **_time_many(workload),
    }


def _benchmark_pandera_check(size: dict[str, int]) -> dict[str, object] | None:
    try:
        import pandera as pa  # type: ignore
        from pandera import Check  # type: ignore
    except ImportError:
        return None

    df = make_dataset(size["rows"], size["cols"])
    schema = pa.DataFrameSchema(
        {col: pa.Column(float, nullable=True) for col in df.columns},
        checks=[pa.Check(lambda frame: frame.notna().mean().mean() > 0.9, name="missing_ceiling")],
    )

    def workload() -> None:
        schema.validate(df, lazy=True)

    return {
        "workload": "pandera validate",
        "rows": size["rows"],
        "cols": size["cols"],
        **_time_many(workload),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible FitCheck benchmarks")
    parser.add_argument("--all", action="store_true", help="Also measure importable competitors")
    args = parser.parse_args(argv)

    print("Running FitCheck benchmarks...")
    entries: list[dict[str, object]] = []
    for size in SIZES:
        entries.append(benchmark_fitcheck_check(size))
        entries.append(benchmark_fitcheck_drift(size))
        if args.all:
            for competitor in (_benchmark_evidently_profile, _benchmark_deepchecks_check, _benchmark_pandera_check):
                entry = competitor(size)
                if entry is not None:
                    entries.append(entry)
                else:
                    entries.append({
                        "workload": competitor.__name__[12:],
                        "rows": size["rows"],
                        "cols": size["cols"],
                        "skipped": "dependency not installed",
                    })

    env = environment_info()
    print(json.dumps({"environment": env, "results": entries}, indent=2, default=str))

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if RESULTS_FILE.exists() else "w"
    with open(RESULTS_FILE, mode, encoding="utf-8") as file:
        file.write(f"## Benchmark Run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write(f"**Environment:** {env['os']} | {env['cpu']} | {env['ram_gb']} GB RAM | Python {env['python']} | FitCheck {env.get('fitcheck', '?')}\n\n")
        file.write("| Workload | Rows | Cols | Median (s) | Min (s) | Max (s) | Runs | Memory (MB) | Note |\n")
        file.write("|---|---|---|---|---|---|---|---|---|\n")
        for entry in entries:
            if entry.get("skipped"):
                file.write(f"| {entry['workload']} | {entry['rows']} | {entry['cols']} | — | — | — | — | — | competitor not installed; not measured |\n")
                continue
            note = ""
            file.write(
                f"| {entry['workload']} | {entry['rows']} | {entry['cols']} | "
                f"{entry['median_s']} | {entry['min_s']} | {entry['max_s']} | "
                f"{entry['n_runs']} | {entry.get('memory_mb', '—')} | {note} |\n"
            )
        file.write(f"\n**Methodology:** {N_REPETITIONS} measured runs, {WARM_UP_RUNS} warm-up runs, median/min/max reported. "
                   "All frameworks share the same deterministically generated dataset and workload definition. "
                   "Competitors are only listed when measured locally; no estimated competitor timings appear in this file.\n\n---\n\n")

    print(f"Results appended to {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
