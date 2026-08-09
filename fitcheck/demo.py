"""FitCheck v2.0 — Quick Demo

Generates all 3 report types in one run:
1. Dataset health check (with intentional issues)
2. Model evaluation (classification)
3. Drift detection

Usage:
    python demo.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import fitcheck

print("=" * 60)
print("FitCheck v2.0 — Demo")
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
df.to_csv("demo_data.csv", index=False)
print(f"    Saved: demo_data.csv ({len(df)} rows, {len(df.columns)} columns)")

# Run check
print("\n[1/3] Running dataset health check...")
issues = fitcheck.check(
    "demo_data.csv", target="label", output="demo_check_report.html", auto_fix=True
)
print(f"    Issues found: {len(issues)}")
print("    Report: demo_check_report.html")
print("    Fix script: demo_check_report_fix_script.py")

# 2. Model evaluation
print("\n[2/3] Training model and evaluating...")
clean_df = df.dropna()
X = clean_df.drop(columns=["label", "constant_col"])
y = clean_df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
metrics = fitcheck.report(model, X_test, y_test, output="demo_model_report.html")
print(f"    Accuracy: {metrics['accuracy']:.4f}")
print(f"    F1 Score: {metrics['f1']:.4f}")
print("    Report: demo_model_report.html")

# 3. Drift detection
print("\n[3/3] Detecting distribution drift...")
ref = pd.DataFrame({"feat_a": np.random.normal(0, 1, 300), "feat_b": np.random.normal(5, 2, 300)})
prod = pd.DataFrame(
    {"feat_a": np.random.normal(0, 1, 300), "feat_b": np.random.normal(8, 2, 300)}
)  # shifted mean
results = fitcheck.detect_drift(ref, prod, output="demo_drift_report.html")
drifted = sum(1 for r in results if r["drifted"])
print(f"    Features tested: {len(results)}, Drifted: {drifted}")
print("    Report: demo_drift_report.html")

print("\n" + "=" * 60)
print("Demo complete! Open the HTML reports in your browser.")
print("=" * 60)
