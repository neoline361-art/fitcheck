# FitCheck Benchmarks

All timing figures below are **actual measured results**, produced by the reproducible runner:

```
python benchmarks/run.py          # FitCheck workloads
python benchmarks/run.py --all    # FitCheck + any importable competitors (Evidently, Deepchecks, Pandera)
```

The runner uses a deterministically generated dataset (identical for every framework on every run), performs three warm-up repetitions, then takes **10 measured runs per workload and reports the median, minimum, and maximum**. Competitors appear in the table only when they were measured locally on the same hardware; the repository never publishes estimated competitor timings.

## Measured runs

### 2026-08-18 03:41:17 (Linux 6.1.102, Intel Xeon @ 2.50GHz, 3.8 GB RAM, Python 3.12.3, FitCheck 3.1.3)

| Workload | Rows | Cols | Median (s) | Min (s) | Max (s) | Runs | Memory (MB) |
|---|---|---|---|---|---|---|---|
| fitcheck check | 1000 | 10 | 0.0205 | 0.0196 | 0.0212 | 10 | 0.1 |
| fitcheck drift | 1000 | 10 | 0.0242 | 0.0237 | 0.0249 | 10 | — |
| fitcheck check | 10000 | 20 | 0.0647 | 0.06 | 0.0674 | 10 | 1.7 |
| fitcheck drift | 10000 | 20 | 0.0344 | 0.0334 | 0.0372 | 10 | — |
| fitcheck check | 100000 | 30 | 0.4631 | 0.4447 | 0.4828 | 10 | 24.8 |
| fitcheck drift | 100000 | 30 | 0.2376 | 0.2216 | 0.2463 | 10 | — |

Competitors Evidently, Deepchecks, and Pandera were **not measured** in this environment (packages not installed), so no comparative table is shown. Run `python benchmarks/run.py --all` in an environment where they are installed to extend this file with locally measured values.

### Methodology

Same hardware, OS, Python version, dataset generation (seed 42), workload definition, and measurement method for every framework. Memory figures reflect the dataset plus loader overhead at warm execution, not cold-start import time. Raw per-run JSON is printed by the runner and appended alongside the table in `results.md`.

---
