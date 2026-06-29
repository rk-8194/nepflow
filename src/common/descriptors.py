"""Shared NEP descriptor loading and computation helpers."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from NepTrainKit.core.calculator import NepCalculator

logger = logging.getLogger("nepflow.common.descriptors")


def descriptor_cache_path(project_dir: Path) -> Path:
    """Return the descriptor cache path for a project."""
    return project_dir / "nep" / "datasets" / "descriptors.npy"


def load_or_compute_descriptors(
    project_dir: Path,
    structures: list,
    *,
    mean_descriptor: bool,
    batch_size: int,
    nep_model_file: str,
) -> np.ndarray:
    """Load cached descriptors or compute them from the configured NEP model."""
    descriptor_cache = descriptor_cache_path(project_dir)
    descriptors = None
    if descriptor_cache.exists():
        descriptors = np.load(descriptor_cache)
        logger.info(f"  Loaded cached descriptors from {descriptor_cache}")
        if descriptors.shape[0] != len(structures):
            logger.warning(
                "  Cache mismatch: %d descriptors vs %d structures - recomputing",
                descriptors.shape[0],
                len(structures),
            )
            descriptors = None

    if descriptors is not None:
        return descriptors

    nep_model_path = project_dir / "config" / "nep" / nep_model_file
    if not nep_model_path.exists():
        raise FileNotFoundError(
            f"NEP model not found at {nep_model_path}\n"
            f"Download NEP89 from:\n"
            f"  https://github.com/brucefan1983/GPUMD/tree/master/potentials/nep/nep89_20250409\n"
            f"Place the model file as: {nep_model_path}"
        )

    calc = NepCalculator(str(nep_model_path))
    logger.info(f"  Loaded NEP model: {nep_model_file}")
    descriptors = compute_descriptors_batched(
        calc,
        structures,
        mean_descriptor=mean_descriptor,
        batch_size=batch_size,
    )
    descriptor_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(descriptor_cache, descriptors)
    logger.info(f"  Saved descriptors to {descriptor_cache}")
    return descriptors


def compute_descriptors_batched(
    calc: NepCalculator,
    structures: list,
    *,
    mean_descriptor: bool,
    batch_size: int,
) -> np.ndarray:
    """Compute descriptors in batches to avoid OOM."""
    all_descriptors = []
    n = len(structures)
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = structures[start:end]
        desc = calc.get_structures_descriptor(
            batch,
            mean_descriptor=mean_descriptor,
        )
        all_descriptors.append(desc)
        elapsed = time.perf_counter() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta = (n - end) / rate if rate > 0 else 0
        logger.info(
            f"  Batch {end}/{n} ({100 * end / n:.0f}%) - "
            f"{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining"
        )
    return np.concatenate(all_descriptors, axis=0)
