"""Deterministic debug selection path."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from .artifacts import write_selected_structures
from .models import SelectionResult


logger = logging.getLogger("nepflow.selection.debug")


def run_debug_selection(
    project_dir: Path,
    ase_structures: list | None = None,
) -> SelectionResult:
    """Create the established deterministic random debug split."""

    generated_path = (
        project_dir / "structures" / "generated" / "generated_structures.xyz"
    )
    if not generated_path.exists():
        raise FileNotFoundError(
            f"No generated structures found at {generated_path}\n"
            "Run the 'generate' stage first."
        )

    if ase_structures is None:
        ase_structures = ase_read(str(generated_path), index=":", format="extxyz")
    if not isinstance(ase_structures, list):
        ase_structures = [ase_structures]
    n = len(ase_structures)
    logger.info("[DEBUG] Loaded %d structures from %s", n, generated_path)

    rng = np.random.RandomState(42)
    representations = rng.randn(n, 10).astype(np.float64)
    logger.info(
        "[DEBUG] Generated in-memory random descriptors (%s); no scientific "
        "descriptor cache is written",
        representations.shape,
    )

    indices = rng.permutation(n)
    n_train = max(1, int(round(0.7 * n)))
    train_indices = sorted(indices[:n_train].tolist())
    test_indices = sorted(indices[n_train:].tolist())
    logger.info(
        "[DEBUG] Random split: %d train, %d test",
        len(train_indices),
        len(test_indices),
    )
    write_selected_structures(
        project_dir,
        ase_structures,
        train_indices,
        test_indices,
    )
    logger.info("[DEBUG] Structure selection complete")
    return SelectionResult(
        descriptors=representations,
        train_indices=train_indices,
        train_min_dist=0.0,
        train_seed_count=0,
        train_single_element_elastic_count=0,
        train_elastic_count=0,
        train_anchor_count=0,
        train_fps_count=len(train_indices),
        test_indices=test_indices,
        test_min_dist=0.0,
        min_train_test_dist=0.0,
        mean_train_test_dist=0.0,
        seed_indices=[],
        single_element_elastic_indices=[],
        elastic_indices=[],
    )


__all__ = ["run_debug_selection"]
