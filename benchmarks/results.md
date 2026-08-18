# FitCheck Benchmarks

Run `python benchmarks/run.py` to reproduce these results.

## check

| Dataset | Time (s) | Memory (MB) |
|---------|----------|-------------|
| 1000 rows x 10 cols | 0.082 | 0.2 |
| 10000 rows x 20 cols | 0.156 | 3.1 |
| 100000 rows x 30 cols | 0.423 | 42.5 |

## drift

| Dataset | Time (s) |
|---------|----------|
| 1000 rows x 10 cols | 0.095 |
| 10000 rows x 20 cols | 0.184 |
| 100000 rows x 30 cols | 0.512 |

*Hardware: Intel i7-12700H, 32GB DDR5, Ubuntu 24.04, Python 3.12*

---
## Benchmark Run: 2026-08-18 03:41:17

**Environment:** Linux 6.1.102 | Intel(R) Xeon(R) Processor @ 2.50GHz | 3.8 GB RAM | Python 3.12.3 | FitCheck 3.1.3

| Workload | Rows | Cols | Median (s) | Min (s) | Max (s) | Runs | Memory (MB) | Note |
|---|---|---|---|---|---|---|---|---|
| fitcheck check | 1000 | 10 | 0.0205 | 0.0196 | 0.0212 | 10 | 0.1 |  |
| fitcheck drift | 1000 | 10 | 0.0242 | 0.0237 | 0.0249 | 10 | — |  |
| fitcheck check | 10000 | 20 | 0.0647 | 0.06 | 0.0674 | 10 | 1.7 |  |
| fitcheck drift | 10000 | 20 | 0.0344 | 0.0334 | 0.0372 | 10 | — |  |
| fitcheck check | 100000 | 30 | 0.4631 | 0.4447 | 0.4828 | 10 | 24.8 |  |
| fitcheck drift | 100000 | 30 | 0.2376 | 0.2216 | 0.2463 | 10 | — |  |

**Methodology:** 10 measured runs, 3 warm-up runs, median/min/max reported. All frameworks share the same deterministically generated dataset and workload definition. Competitors are only listed when measured locally; no estimated competitor timings appear in this file.

---

