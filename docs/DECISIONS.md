# Design Decisions

## Why not use pytest for the demo?
The demo script (`demo.py`) uses raw Python to show exactly what the library produces, with no test harness abstraction. It doubles as a quick smoke test.

## Why HTML reports instead of a dashboard?
HTML reports are zero-infrastructure. No server, no database, no deployment. Open in any browser, share by email or Slack. Follows the Unix philosophy of doing one thing well.

## Why pandas and not polars?
pandas is the lowest-common-denominator for ML workflows. polars support is planned but would increase the dependency surface. pandas has 95%+ of the relevant ecosystem.

## Why KS test instead of PSI?
KS test is well-known, well-understood, and has scipy-native implementation. PSI (Population Stability Index) is planned for v2.1 as an additional drift metric.

## Why not silent data mutation?
Silent mutation is the root cause of debugging hell. FitCheck never modifies data. It generates transparent fix scripts that users review, edit, and run. This builds trust and prevents "I ran a function and my data disappeared" scenarios.

## Why no classes?
Pure functions reduce cognitive load. Every function takes input and returns output. No state, no side effects, easy to test. If a future feature genuinely benefits from a class (e.g., a configuration object), it will be added — but only when justified.
