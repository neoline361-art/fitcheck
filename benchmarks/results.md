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
