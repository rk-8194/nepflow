#!/usr/bin/env python3
"""
Compute NEP descriptors from a trained potential and plot descriptor space.

Uses a trained nep.txt potential (from a completed train_nep run) instead of
the NEP89 foundation model. Mirrors the descriptor computation and plot from
the select stage, but adds a fourth colour layer for the NEP training data.

Plot layers (back to front):
  gray   - unselected (not in FPS train/test selection)
  red    - FPS-selected train (structures/selected/train.xyz)
  blue   - FPS-selected test  (structures/selected/test.xyz)
  green  - structures actually used in NEP training (potential/train.xyz)

Usage:
    python plot_descriptors.py --project PATH [--potential NAME] [options]

Arguments:
    --project   PATH   Path to the nepflow project directory
    --potential NAME   Potential folder inside nep/potentials/ or "latest"
                       (default: "latest")
    --structures PATH  XYZ file to describe
                       (default: structures/generated/generated_structures.xyz)
    --train-xyz PATH   FPS-selected train set XYZ (default: structures/selected/train.xyz)
    --test-xyz  PATH   FPS-selected test set XYZ  (default: structures/selected/test.xyz)
    --output    PATH   Output PNG path (default: reports/descriptor_space_trained.png)
    --batch     INT    Descriptor batch size (default: 500)
    --no-cache         Recompute descriptors even if a cache exists
    --atom             Use per-atom descriptors instead of per-structure mean
"""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plot_descriptors")


# ---------------------------------------------------------------------------
# helpers — copied / adapted from select.py
# ---------------------------------------------------------------------------

def _latest_potential(potentials_dir: Path) -> Path:
    candidates = [
        d for d in potentials_dir.iterdir()
        if d.is_dir() and (d / "nep.txt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed potential folders found in {potentials_dir}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _structure_key(atoms) -> str:
    """Stable hash: formula + rounded positions."""
    pos_bytes = atoms.get_positions().round(6).tobytes()
    return hashlib.md5(atoms.get_chemical_formula().encode() + pos_bytes).hexdigest()


def _match_indices(ref_ase: list, target_xyz: Path) -> list[int]:
    """Return indices into ref_ase for structures in target_xyz."""
    if not target_xyz.exists():
        logger.warning(f"  {target_xyz} not found — skipping")
        return []
    target = ase_read(str(target_xyz), index=":", format="extxyz")
    if not isinstance(target, list):
        target = [target]
    ref_index = {_structure_key(a): i for i, a in enumerate(ref_ase)}
    indices = [ref_index[_structure_key(a)] for a in target if _structure_key(a) in ref_index]
    if not indices:
        logger.warning(f"  No structures from {target_xyz.name} matched")
    else:
        logger.info(f"  Matched {len(indices)}/{len(target)} from {target_xyz.name}")
    return indices


def _compute_descriptors_batched(
    calc: NepCalculator,
    structures: list,
    mean_descriptor: bool,
    batch_size: int,
) -> np.ndarray:
    """Compute descriptors in batches to avoid OOM. (Same as select.py.)"""
    all_descriptors = []
    n = len(structures)
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        desc = calc.get_structures_descriptor(structures[start:end], mean_descriptor=mean_descriptor)
        all_descriptors.append(desc)
        elapsed = time.perf_counter() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta = (n - end) / rate if rate > 0 else 0
        logger.info(f"  Batch {end}/{n} ({100*end/n:.0f}%) — {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining")
    return np.concatenate(all_descriptors, axis=0)



def _plot(
    descriptors: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
    nep_train_indices: list[int],
    output_path: Path,
    potential_name: str,
) -> None:
    """PCA 2D scatter.

    Layers (back to front):
      gray   — unselected (not in FPS selection)
      red    — FPS-selected train (structures/selected/train.xyz)
      blue   — FPS-selected test  (structures/selected/test.xyz)
      green  — NEP training data  (potential/train.xyz, subset of red)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    coords = pca.fit_transform(descriptors)

    n = len(descriptors)
    train_mask     = np.zeros(n, dtype=bool)
    test_mask      = np.zeros(n, dtype=bool)
    nep_train_mask = np.zeros(n, dtype=bool)
    if train_indices:
        train_mask[train_indices] = True
    if test_indices:
        test_mask[test_indices] = True
    if nep_train_indices:
        nep_train_mask[nep_train_indices] = True
    unselected = ~(train_mask | test_mask)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        coords[unselected, 0], coords[unselected, 1],
        s=4, alpha=0.2, c="gray",
        label=f"Unselected ({unselected.sum()})",
    )
    if train_mask.any():
        ax.scatter(
            coords[train_mask, 0], coords[train_mask, 1],
            s=10, alpha=0.4, c="tab:red",
            label=f"FPS-selected train ({train_mask.sum()})",
        )
    if test_mask.any():
        ax.scatter(
            coords[test_mask, 0], coords[test_mask, 1],
            s=10, alpha=0.7, c="tab:blue",
            label=f"FPS-selected test ({test_mask.sum()})",
        )
    if nep_train_mask.any():
        ax.scatter(
            coords[nep_train_mask, 0], coords[nep_train_mask, 1],
            s=18, alpha=0.9, c="tab:green",
            label=f"NEP training data ({nep_train_mask.sum()})",
        )

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title(f"NEP Descriptor Space — Trained potential: {potential_name}")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute descriptors with a trained NEP potential and plot descriptor space."
    )
    parser.add_argument("--project", type=Path, required=True,
                        help="Path to the nepflow project directory")
    parser.add_argument("--potential", default="latest",
                        help='Potential folder inside nep/potentials/ or "latest" (default: latest)')
    parser.add_argument("--structures", type=Path, default=None,
                        help="XYZ file to describe (default: structures/generated/generated_structures.xyz)")
    parser.add_argument("--train-xyz", type=Path, default=None,
                        help="FPS-selected train set (default: structures/selected/train.xyz)")
    parser.add_argument("--test-xyz", type=Path, default=None,
                        help="FPS-selected test set (default: structures/selected/test.xyz)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: reports/descriptor_space_trained.png)")
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

    # Resolve potential
    potentials_dir = project_dir / "nep" / "potentials"
    if args.potential == "latest":
        potential_path = _latest_potential(potentials_dir)
        logger.info(f"Using latest potential: {potential_path.name}")
    else:
        potential_path = potentials_dir / args.potential
        if not potential_path.exists():
            logger.error(f"Potential folder not found: {potential_path}")
            sys.exit(1)

    nep_txt = potential_path / "nep.txt"
    if not nep_txt.exists():
        logger.error(f"nep.txt not found in {potential_path} — has training completed?")
        sys.exit(1)
    logger.info(f"Using potential: {nep_txt}")

    structures_path = args.structures or (project_dir / "structures" / "generated" / "generated_structures.xyz")
    train_xyz       = args.train_xyz  or (project_dir / "structures" / "selected" / "train.xyz")
    test_xyz        = args.test_xyz   or (project_dir / "structures" / "selected" / "test.xyz")
    output_path     = args.output     or (project_dir / "reports" / "descriptor_space_trained.png")
    mean_descriptor = not args.atom

    # ----------------------------------------------------------
    # Step 1: Load structures  (same as select.py)
    # ----------------------------------------------------------
    logger.info("")
    logger.info("Step 1: Loading generated structures")
    t0 = time.perf_counter()
    structures = Structure.read_multiple(str(structures_path))
    ase_structures = ase_read(str(structures_path), index=":", format="extxyz")
    if not isinstance(ase_structures, list):
        ase_structures = [ase_structures]
    logger.info(f"  Loaded {len(structures)} structures ({time.perf_counter()-t0:.1f}s)")

    if len(structures) == 0:
        logger.error("No structures found")
        sys.exit(1)

    # ----------------------------------------------------------
    # Step 2: Load or compute descriptors  (same as select.py)
    # ----------------------------------------------------------
    logger.info("")
    logger.info("Step 2: Computing NEP descriptors")

    cache_path = potential_path / "descriptors_cache.npy"
    descriptors = None

    if cache_path.exists() and not args.no_cache:
        descriptors = np.load(cache_path)
        logger.info(f"  Loaded cached descriptors from {cache_path}")
        if descriptors.shape[0] != len(structures):
            logger.warning(
                f"  Cache mismatch: {descriptors.shape[0]} vs {len(structures)} structures — recomputing"
            )
            descriptors = None

    if descriptors is None:
        calc = NepCalculator(str(nep_txt))
        logger.info(f"  Loaded NepCalculator with {nep_txt.name}")
        descriptors = _compute_descriptors_batched(calc, structures, mean_descriptor, args.batch)
        np.save(cache_path, descriptors)
        logger.info(f"  Saved descriptor cache to {cache_path}")

    logger.info(f"  Descriptor shape: {descriptors.shape}")

    # ----------------------------------------------------------
    # Step 3: Match train / test / nep-train indices
    # ----------------------------------------------------------
    logger.info("")
    logger.info("Step 3: Matching train/test/NEP-training sets")
    train_indices     = _match_indices(ase_structures, train_xyz)
    test_indices      = _match_indices(ase_structures, test_xyz)
    nep_train_xyz     = potential_path / "train.xyz"
    nep_train_indices = _match_indices(ase_structures, nep_train_xyz)
    if not nep_train_indices:
        logger.warning(
            f"  No NEP training structures matched from {nep_train_xyz}\n"
            f"  (train.xyz is copied here by submit_training_job — "
            f"check that it exists in {potential_path.name})"
        )

    # ----------------------------------------------------------
    # Step 4: Save descriptor vectors
    # ----------------------------------------------------------
    logger.info("")
    logger.info("Step 4: Saving descriptor vectors")

    n = len(descriptors)
    labels = np.full(n, "unselected", dtype=object)
    for i in train_indices:
        labels[i] = "train"
    for i in test_indices:
        labels[i] = "test"
    for i in nep_train_indices:
        labels[i] = "nep_train"

    descriptors_out = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        descriptors_out,
        descriptors=descriptors,
        labels=labels,
        train_indices=np.array(train_indices, dtype=int),
        test_indices=np.array(test_indices, dtype=int),
        nep_train_indices=np.array(nep_train_indices, dtype=int),
    )
    logger.info(f"  Saved to {descriptors_out}")

    # ----------------------------------------------------------
    # Step 5: Plot  (same style as select.py)
    # ----------------------------------------------------------
    logger.info("")
    logger.info("Step 5: Plotting descriptor space")
    _plot(descriptors, train_indices, test_indices, nep_train_indices, output_path, potential_path.name)

    logger.info("")
    logger.info("Done.")
    logger.info(f"  Descriptors : {descriptors_out}")
    logger.info(f"  Plot        : {output_path}")


if __name__ == "__main__":
    main()
