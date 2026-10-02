#!/usr/bin/env python3
"""Temporary compatibility entry point for the package CLI."""

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from nepflow.cli import main


if __name__ == "__main__":
    main()
