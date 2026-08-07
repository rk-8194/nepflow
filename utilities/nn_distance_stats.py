#!/usr/bin/env python3
"""
Compute nearest-neighbor distance statistics in NEP descriptor space.

This is a simple coverage diagnostic for a trained NEP potential.
It loads descriptors for a structure set, computes each point's nearest
neighbor distance to the rest of the set in the original descriptor space,
and reports summary statistics including the 95th percentile.

Usage:
    python nn_distance_stats.py --project PATH --potential NAME [options]

Arguments:
    --project   PATH   Path to the nepflow project directory
    --potential NAME   Potential folder inside nep/potentials/ or "latest"
                       (default: "latest")
    --structures PATH  XYZ file to analyze
                       (default: structures/generated/generated_structures.xyz)
    --batch     INT    Descriptor batch size (default: 500)
    --no-cache         Recompute descriptors even if a cache exists
    --atom             Use per-atom descriptors instead of per-structure mean
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.structure import Structure
from scipy.spatial.distance import cdist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nn_distance_stats")


def _compute_structure_descriptors(
    calc: NepCalculator,
    structures: list,
    *,
    mean_descriptor: bool,
) -> np.ndarray:
    if hasattr(calc, "descriptors"):
        return calc.descriptors(structures, mean=mean_descriptor)
    if hasattr(calc, "get_structures_descriptor"):
        return calc.get_structures_descriptor(
            structures,
            mean_descriptor=mean_descriptor,
        )
    raise AttributeError(
        "NepCalculator does not provide a supported descriptor API. "
        "Expected descriptors() or get_structures_descriptor()."
    )


def _latest_potential(potentials_dir: Path) -> Path:
    candidates = [
        d for d in potentials_dir.iterdir()
        if d.is_dir() and (d / "nep.txt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed potential folders found in {potentials_dir}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _resolve_potential(potentials_dir: Path, potential_name: str) -> Path:
    if potential_name == "latest":
        return _latest_potential(potentials_dir)
    potential_path = potentials_dir / potential_name
    if not potential_path.exists():
        raise FileNotFoundError(f"Potential folder not found: {potential_path}")
    return potential_path


def _structure_key(atoms) -> str:
    pos_bytes = atoms.get_positions().round(6).tobytes()
    return hashlib.md5(atoms.get_chemical_formula().encode() + pos_bytes).hexdigest()


def _compute_descriptors_batched(
    calc: NepCalculator,
    structures: list,
    mean_descriptor: bool,
    batch_size: int,
) -> np.ndarray:
    all_descriptors = []
    n = len(structures)
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        desc = _compute_structure_descriptors(
            calc,
            structures[start:end],
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


def _load_or_compute_descriptors(
    potential_path: Path,
    nep_txt: Path,
    structures: list,
    *,
    mean_descriptor: bool,
    batch_size: int,
    no_cache: bool,
) -> np.ndarray:
    cache_path = potential_path / "descriptors_cache.npy"
    descriptors = None

    if cache_path.exists() and not no_cache:
        descriptors = np.load(cache_path)
        logger.info(f"  Loaded cached descriptors from {cache_path}")
        if descriptors.shape[0] != len(structures):
            logger.warning(
                f"  Cache mismatch: {descriptors.shape[0]} vs {len(structures)} structures - recomputing"
            )
            descriptors = None

    if descriptors is None:
        calc = NepCalculator(str(nep_txt))
        logger.info(f"  Loaded NepCalculator with {nep_txt.name}")
        descriptors = _compute_descriptors_batched(
            calc,
            structures,
            mean_descriptor,
            batch_size,
        )
        np.save(cache_path, descriptors)
        logger.info(f"  Saved descriptor cache to {cache_path}")

    return descriptors


def _nearest_neighbor_distances(descriptors: np.ndarray) -> np.ndarray:
    if len(descriptors) < 2:
        return np.array([], dtype=float)
    dists = cdist(descriptors, descriptors)
    np.fill_diagonal(dists, np.inf)
    return dists.min(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute nearest-neighbor distance statistics for a trained NEP potential."
    )
    parser.add_argument("--project", type=Path, required=True,
                        help="Path to the nepflow project directory")
    parser.add_argument("--potential", default="latest",
                        help='Potential folder inside nep/potentials/ or "latest" (default: latest)')
    parser.add_argument("--structures", type=Path, default=None,
                        help="XYZ file to analyze (default: structures/generated/generated_structures.xyz)")
    parser.add_argument("--batch", type=int, default=500,
                        help="Descriptor batch size (default: 500)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute descriptors even if a cache exists")
    parser.add_argument("--atom", action="store_true",
                        help="Use per-atom descriptors instead of per-structure mean")
    args = parser.parse_args()

    project_dir = args.project
    if not project_dir.exists():
        logger.error(f"Project directory not found: {project_dir}")
        sys.exit(1)
    logger.info(f"Project directory: {project_dir}")

    potentials_dir = project_dir / "nep" / "potentials"
    try:
        potential_path = _resolve_potential(potentials_dir, args.potential)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    nep_txt = potential_path / "nep.txt"
    if not nep_txt.exists():
        logger.error(f"nep.txt not found in {potential_path} - has training completed?")
        sys.exit(1)
    logger.info(f"Using potential: {nep_txt}")

    structures_path = args.structures or (
        project_dir / "structures" / "generated" / "generated_structures.xyz"
    )
    mean_descriptor = not args.atom

    logger.info("")
    logger.info("Step 1: Loading structures")
    t0 = time.perf_counter()
    structures = Structure.read_multiple(str(structures_path))
    ase_structures = ase_read(str(structures_path), index=":", format="extxyz")
    if not isinstance(ase_structures, list):
        ase_structures = [ase_structures]
    logger.info(f"  Loaded {len(structures)} structures ({time.perf_counter() - t0:.1f}s)")

    if len(structures) == 0:
        logger.error("No structures found")
        sys.exit(1)

    logger.info("")
    logger.info("Step 2: Computing NEP descriptors")
    descriptors = _load_or_compute_descriptors(
        potential_path,
        nep_txt,
        structures,
        mean_descriptor=mean_descriptor,
        batch_size=args.batch,
        no_cache=args.no_cache,
    )
    logger.info(f"  Descriptor shape: {descriptors.shape}")

    logger.info("")
    logger.info("Step 3: Computing nearest-neighbor distances")
    nn_dists = _nearest_neighbor_distances(descriptors)
    if len(nn_dists) == 0:
        logger.error("Need at least two structures to compute nearest-neighbor distances")
        sys.exit(1)

    p50 = float(np.percentile(nn_dists, 50))
    p95 = float(np.percentile(nn_dists, 95))
    p99 = float(np.percentile(nn_dists, 99))
    mean = float(nn_dists.mean())
    min_dist = float(nn_dists.min())
    max_dist = float(nn_dists.max())

    logger.info("")
    logger.info("Nearest-neighbor distance summary")
    logger.info(f"  Count : {len(nn_dists)}")
    logger.info(f"  Min   : {min_dist:.6f}")
    logger.info(f"  Mean  : {mean:.6f}")
    logger.info(f"  P50   : {p50:.6f}")
    logger.info(f"  P95   : {p95:.6f}")
    logger.info(f"  P99   : {p99:.6f}")
    logger.info(f"  Max   : {max_dist:.6f}")


if __name__ == "__main__":
    main()
