#!/usr/bin/env python3
"""Temporary compatibility wrapper for the package CLI."""

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from nepflow import cli as _cli
from nepflow.cli import (
    SchedulerError,
    main,
    process_runner,
    scheduler,
)

_resolve_resubmit_command = _cli._resolve_resubmit_command


if __name__ == "__main__":
    main()
