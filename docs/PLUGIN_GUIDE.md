# Writing FitCheck Plugins

Custom checks are plain Python functions. A check takes a pandas DataFrame and
returns a list of issue dictionaries; issues flow into the standard report and
respect the CI exit codes.

## The issue schema

```python
{
    "column": "feature_name",       # column the issue applies to, or "all"
    "type": "my_check",             # machine-readable check id
    "severity": "warning",          # "info", "warning", or "critical"
    "message": "Human-readable problem",   # shown in reports
    "suggestion": "How to fix it",         # shown in reports
}
```

## Registering a plugin

Two ways: register by name in-process, or resolve a dotted module path from the
CLI.

```python
# my_checks.py
import pandas as pd


def check(df: pd.DataFrame) -> list[dict]:
    """Flag columns whose mean is negative."""
    issues = []
    for col in df.select_dtypes(include="number"):
        if float(df[col].mean()) < 0:
            issues.append({
                "column": col,
                "type": "negative_mean",
                "severity": "warning",
                "message": f"{col} has a negative mean",
                "suggestion": "Investigate negative values before modeling",
            })
    return issues
```

### Option A — in-process registration

```python
from fitcheck.plugins import registry

registry.register("negative_mean", check)
fitcheck.check("data.csv", plugins=[load_plugin("negative_mean")])
```

### Option B — dotted module path (CLI)

```bash
fitcheck check data.csv --plugins my_checks
```

The loader looks for a callable named `check`, `plugin`, or `run` inside the
module. A plugin must return a `list`; anything else raises `TypeError`.

## Severity and exit codes

`fitcheck check` maps the worst severity to the exit code: `0` clear, `1`
warnings, `2` critical, `3` runtime error. `--fail-on critical` makes only
critical issues fail a pipeline, so plugins can gate merges the same way the
built-in checks do.

## Testing a plugin

```python
def test_plugin():
    import pandas as pd
    from my_checks import check

    df = pd.DataFrame({"x": [-1, -2, 3]})
    assert check(df)[0]["type"] == "negative_mean"
```

## Notes

- Plugins never mutate the input frame; FitCheck passes a copy.
- Keep checks fast: they run per row-set, so vectorize with pandas operations.
- Optional heavy dependencies belong in plugins, not the core package.
