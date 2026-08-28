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

## Can someone fake a FitCheck report?
Not without detection. Every report embeds a visible SHA-256 fingerprint of the source dataset. If the report is edited, `fitcheck verify` will detect the mismatch. With HMAC signing (`--sign-key`), even recomputing the hash is impossible without the secret key.

## How do I verify a report?
Run `fitcheck verify report.html --against data.csv`. It compares the embedded fingerprint against the current file hash. With HMAC signing, it also validates the cryptographic signature.

## Why is FitCheck open source if you want it to be trustworthy?
Trust comes from verifiability, not obscurity. Closed-source tools ask you to trust the vendor. FitCheck lets you verify every report cryptographically — the code is Apache 2.0 licensed, same as TensorFlow and PyTorch.

## What is HMAC signing and do I need it?
HMAC-SHA256 signing adds a cryptographic signature to reports that can only be produced with a secret key. It prevents someone from regenerating a valid fingerprint after tampering. Use it in CI pipelines (`--sign-key $SECRET`) or via the `FITCHECK_SECRET_KEY` environment variable. It's optional — unsigned reports still have SHA-256 fingerprints.
