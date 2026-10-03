"""Compatibility adapters for the canonical training dataset boundary.

New callers use :mod:`nepflow.stages.training.dataset`.  These small adapters
remain only for older integrations while their implementation is owned by the
canonical package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ase.atoms import Atoms

from nepflow.stages.training.dataset import (
    DatasetBuildReport,
    DatasetSplit,
    iter_labeled_structures,
    write_nep_dataset,
)


def prepare_dataset(
    dataset_path: Path,
    ase_structures: Sequence[Atoms],
    is_train: bool,
    project_dir: Path,
    train_virial: bool = False,
    debug: bool = False,
    extraction_report: dict | None = None,
) -> int:
    """Deprecated single-split adapter for pre-Phase 4 callers."""

    split = DatasetSplit.TRAIN if is_train else DatasetSplit.TEST
    report = DatasetBuildReport(split, requested_count=len(ase_structures))
    structures = list(
        iter_labeled_structures(
            ase_structures,
            split,
            project_dir,
            require_virial=train_virial,
            debug=debug,
            report=report,
        )
    )
    count = write_nep_dataset(
        dataset_path,
        structures,
        include_virial=train_virial,
    )
    if extraction_report is not None:
        extraction_report.clear()
        extraction_report.update(report.to_dict())
    return count


__all__ = ["prepare_dataset"]
