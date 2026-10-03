"""Persistent selection artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

from ase.io import write as ase_write


logger = logging.getLogger("nepflow.selection.artifacts")


def write_selected_structures(
    project_dir: Path,
    ase_structures: list,
    train_indices: list[int],
    test_indices: list[int],
) -> tuple[Path, Path]:
    """Write selected train/test structures using the established paths."""

    selected_dir = project_dir / "structures" / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)

    train_path = selected_dir / "train.xyz"
    ase_write(
        str(train_path),
        [ase_structures[index] for index in train_indices],
        format="extxyz",
    )
    logger.info("  Training set saved to %s", train_path)

    test_path = selected_dir / "test.xyz"
    ase_write(
        str(test_path),
        [ase_structures[index] for index in test_indices],
        format="extxyz",
    )
    logger.info("  Test set saved to %s", test_path)
    return train_path, test_path


__all__ = ["write_selected_structures"]
