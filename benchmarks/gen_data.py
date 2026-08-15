"""Generate a synthetic 100k-row dataset for local benchmarking.

Usage:
    python benchmarks/gen_data.py

Writes ``test_100k.csv`` with realistic issue patterns: missing values in
``age``, imbalanced classes, a skewed categorical column, and a constant-like
``city`` field. This is a local test-data generator, not part of the package.
"""

import numpy as np
import pandas as pd


def generate() -> None:
    np.random.seed(42)
    n = 100000
    age = np.random.normal(35, 12, n).astype(np.float32)
    age[np.random.rand(n) < 0.03] = np.nan
    income = np.random.lognormal(10.8, 0.6, n).astype(np.float32)
    credit = np.random.normal(680, 80, n).astype(np.float32)
    city = np.random.choice(["NYC", "LA", "Chicago"], n)
    city[:50000] = "NYC"
    dept = np.random.choice(["Eng", "Sales", "Marketing"], n, p=[0.6, 0.3, 0.1])
    edu = np.random.choice(["HS", "Bachelor", "Master"], n, p=[0.2, 0.5, 0.3])
    active = np.random.choice([0, 1], n, p=[0.15, 0.85]).astype(np.float32)
    churn = (np.random.rand(n) < 0.05).astype(int)
    df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "age": age,
            "income": income,
            "credit_score": credit,
            "city": city,
            "department": dept,
            "education": edu,
            "is_active": active,
            "churn": churn,
        }
    )
    df.to_csv("test_100k.csv", index=False)
    print(f"Created test_100k.csv: {len(df)} rows")


if __name__ == "__main__":
    generate()
