"""Enable `python -m fitcheck`."""

from __future__ import annotations

import sys

from fitcheck.cli import main

if __name__ == "__main__":
    sys.exit(main())
