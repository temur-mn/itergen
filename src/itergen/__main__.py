"""Module execution stub for API-only vorongen package."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "itergen no longer provides a CLI. "
        "Use the Python API (see README) or run sample_run.py.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
