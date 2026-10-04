"""NEP progress and backend-error interpretation."""

from __future__ import annotations

import math
import re
from pathlib import Path

from nepflow.errors import MlipError
from nepflow.mlip.backend import TrainingProgress


def parse_progress_line(line: str) -> TrainingProgress | None:
    """Parse the generation and total loss from one ``loss.out`` line."""

    fields = line.split()
    if len(fields) < 2:
        return None
    try:
        generation = int(fields[0])
        loss = float(fields[1])
        if generation < 0 or not math.isfinite(loss):
            return None
        return TrainingProgress(generation, loss)
    except (TypeError, ValueError, MlipError):
        return None


def parse_progress(run_directory: Path) -> TrainingProgress | None:
    """Read the last valid progress line from NEP's ``loss.out``."""

    source = Path(run_directory) / "loss.out"
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        result = parse_progress_line(line.strip())
        if result is not None:
            return result
    return None


def classify_training_error(run_directory: Path) -> str | None:
    """Return a stable backend error category from the training log, if any."""

    directory = Path(run_directory)
    log_paths = sorted(directory.glob("train_nep_*.log"))
    if not log_paths:
        return None
    try:
        content = log_paths[0].read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lowered = content.lower()
    patterns = (
        ("out_of_memory", ("out of memory", "oom", "cuda out of memory")),
        ("cuda_error", ("cuda error", "cuda failure", "no cuda device")),
        ("segmentation_fault", ("segmentation fault", "sigsegv")),
        ("killed", ("killed", "terminated")),
        ("nonconvergence", ("not converged", "non-convergence", "nonconvergence")),
    )
    for category, needles in patterns:
        if any(needle in lowered for needle in needles):
            return category
    for line in reversed(content.splitlines()):
        if re.search(r"\b(error|failed|failure)\b", line, re.IGNORECASE):
            return "backend_error"
    return None


__all__ = ["classify_training_error", "parse_progress", "parse_progress_line"]
