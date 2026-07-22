# FAQ

## Does FitCheck modify my data?
No. FitCheck diagnoses, never mutates. `auto_fix=True` generates a Python script — you review it before running.

## What file formats are supported?
CSV and Parquet.

## Do I need an internet connection?
No. Everything runs locally.

## Can I use FitCheck with deep learning models?
Currently only scikit-learn-compatible models. Deep learning support is planned.

## What if my dataset is very large?
FitCheck loads the full dataset into memory. For datasets larger than available RAM, consider sampling or chunked processing.

## Does FitCheck send data anywhere?
No. Zero telemetry, zero outbound network calls.

## How do I cite FitCheck?
```
FitCheck: Zero-boilerplate ML data validation. https://github.com/neoline361-art/fitcheck
```

## Can I contribute?
Yes. See CONTRIBUTING.md.
