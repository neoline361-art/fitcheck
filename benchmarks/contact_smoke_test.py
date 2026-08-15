from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from fitcheck.check import check

out = Path("/tmp/fitcheck-contact-smoke")
out.mkdir(exist_ok=True)
path = out / "contacts_1m.csv"
rows = 1_000_000
if not path.exists():
    frame = pd.DataFrame(
        {
            "mobile": [f"+1202555{i:05d}" for i in range(rows)],
            "name": [f"Person {i:07d}" for i in range(rows)],
        }
    )
    frame.to_csv(path, index=False)

start = time.perf_counter()
full = check(path.as_posix(), output=(out / "full_report.html").as_posix(), return_format="dict")
full_seconds = time.perf_counter() - start
start = time.perf_counter()
sampled = check(path.as_posix(), sample_rows=100_000, output=(out / "sample_report.html").as_posix(), return_format="dict")
sample_seconds = time.perf_counter() - start
print({"path": str(path), "full_rows": full["total_rows"], "full_seconds": round(full_seconds, 2), "sample_rows": sampled["total_rows"], "sample_seconds": round(sample_seconds, 2), "full_report": str(out / "full_report.html"), "sample_report": str(out / "sample_report.html")})
