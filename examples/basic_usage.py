"""FitCheck basic usage examples. Run with: python examples/basic_usage.py"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import fitcheck


def example_check():
    """Validate a dataset's health."""
    df = pd.DataFrame({
        "age": [25, 30, 35, None, 40, 45, 50, 55, None, 28],
        "income": [50000, 60000, 55000, 70000, 65000, 48000, 52000, 58000, 62000, 51000],
        "department": ["eng", "eng", "sales", "sales", "eng", "eng", "sales", "eng", "sales", "eng"],
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })
    issues = fitcheck.check(df, target="target", output="/tmp/fitcheck_check_example.html")
    print(f"Check: {len(issues)} issues found")


def example_report():
    """Evaluate a trained model."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=25, random_state=42)
    model.fit(X_train, y_train)
    metrics = fitcheck.report(model, X_test, y_test, output="/tmp/fitcheck_model_example.html")
    print(f"Report: accuracy={metrics['accuracy']:.3f}")


def example_drift():
    """Detect distribution drift."""
    ref = pd.DataFrame({"feat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    prod = pd.DataFrame({"feat": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]})
    results = fitcheck.detect_drift(ref, prod, output="/tmp/fitcheck_drift_example.html")
    print(f"Drift: {sum(r['drifted'] for r in results)} drifted features")


if __name__ == "__main__":
    example_check()
    example_report()
    example_drift()
    print("Examples complete. Reports in /tmp/")
