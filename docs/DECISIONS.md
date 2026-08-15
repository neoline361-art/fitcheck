# Design Decisions

## Why not use pytest for the demo?
The demo script (`demo.py`) uses raw Python to show exactly what the library produces, with no test harness abstraction. It doubles as a quick smoke test.

## Why HTML reports instead of a dashboard?
HTML reports are zero-infrastructure. No server, no database, no deployment. Open in any browser, share by email or Slack. Follows the Unix philosophy of doing one thing well.

## Why pandas and not polars?
pandas is the lowest-common-denominator for ML workflows; the check engine is pandas-native. Polars is available as an optional loading backend (`--backend polars`) for large files — it accelerates the load step and the frames convert to pandas for checks. Polars-native checks are a future optimisation, not a current requirement.

## Why KS and PSI both?
KS is a well-known hypothesis test with a scipy-native implementation; PSI measures population shift on a fixed scale and is the industry default for credit risk. Auto-selection picks KS for smaller numeric samples and PSI for larger ones, with Wasserstein and Jensen–Shannon available explicitly.

## Why not silent data mutation?
Silent mutation is the root cause of debugging hell. FitCheck never modifies data. It generates transparent fix scripts that users review, edit, and run. This builds trust and prevents "I ran a function and my data disappeared" scenarios.

## Why no classes?
Pure functions reduce cognitive load. Every function takes input and returns output. No state, no side effects, easy to test. If a future feature genuinely benefits from a class (e.g., a configuration object), it will be added — but only when justified.
