"""Deterministic debug selection path."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from .artifacts import write_selected_structures
from .representations import descriptor_cache_path


logger = logging.getLogger("nepflow.selection.debug")


def run_debug_selection(project_dir: Path) -> None:
    """Create the established deterministic random debug split."""

    generated_path = (
        project_dir / "structures" / "generated" / "generated_structures.xyz"
    )
    if not generated_path.exists():
        raise FileNotFoundError(
            f"No generated structures found at {generated_path}\n"
            "Run the 'generate' stage first."
        )

    ase_structures = ase_read(str(generated_path), index=":", format="extxyz")
    n = len(ase_structures)
    logger.info("[DEBUG] Loaded %d structures from %s", n, generated_path)

    rng = np.random.RandomState(42)
    representations = rng.randn(n, 10).astype(np.float64)
    descriptor_cache = descriptor_cache_path(project_dir)
    descriptor_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(descriptor_cache, representations)
    logger.info("[DEBUG] Saved random descriptors (%s) to %s", representations.shape, descriptor_cache)

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


__all__ = ["run_debug_selection"]
