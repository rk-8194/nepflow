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
    python plot_descriptors.py --project PATH --compare-potential NAME --shared-pca [options]

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
    --compare-potential NAME
                       Second potential folder inside nep/potentials/ for shared-PCA comparison
    --shared-pca       Fit one PCA basis on both potentials and plot them side by side
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plot_descriptors")


# ---------------------------------------------------------------------------
# helpers - copied / adapted from select.py
# ---------------------------------------------------------------------------

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


def _structure_key(atoms) -> str:
    """Stable hash: formula + rounded positions."""
    pos_bytes = atoms.get_positions().round(6).tobytes()
    return hashlib.md5(atoms.get_chemical_formula().encode() + pos_bytes).hexdigest()


def _match_indices(ref_ase: list, target_xyz: Path) -> list[int]:
    """Return indices into ref_ase for structures in target_xyz."""
    if not target_xyz.exists():
        logger.warning(f"  {target_xyz} not found - skipping")
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


def _resolve_potential(potentials_dir: Path, potential_name: str) -> Path:
    if potential_name == "latest":
        return _latest_potential(potentials_dir)
    potential_path = potentials_dir / potential_name
    if not potential_path.exists():
        raise FileNotFoundError(f"Potential folder not found: {potential_path}")
    return potential_path


def _resolve_compare_target(
    project_dir: Path,
    potentials_dir: Path,
    compare_name: str,
) -> tuple[Path, Path, str]:
    """Return (cache_dir, nep_txt_path, label) for a comparison target."""
    normalized = compare_name.strip()
    if normalized in {"default", "nep89", "nep89.txt"}:
        nep_txt = project_dir / "config" / "nep" / "nep89.txt"
        if not nep_txt.exists():
            raise FileNotFoundError(f"Default NEP model not found: {nep_txt}")
        return nep_txt.parent, nep_txt, nep_txt.name

    candidate = Path(normalized)
    if candidate.suffix == ".txt":
        if not candidate.is_absolute():
            if candidate.exists():
                pass
            elif (project_dir / candidate).exists():
                candidate = project_dir / candidate
            else:
                candidate = project_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Comparison model file not found: {candidate}")
        return candidate.parent, candidate, candidate.stem

    potential_path = _resolve_potential(potentials_dir, normalized)
    nep_txt = potential_path / "nep.txt"
    if not nep_txt.exists():
        raise FileNotFoundError(f"nep.txt not found in {potential_path}")
    return potential_path, nep_txt, potential_path.name


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


def _fit_pca(descriptors: np.ndarray):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(descriptors)
    return pca


def _project_with_pca(descriptors: np.ndarray, pca) -> np.ndarray:
    return pca.transform(descriptors)


def _plot_panel(
    ax,
    coords: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
    nep_train_indices: list[int],
    title: str,
) -> None:
    n = len(coords)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    nep_train_mask = np.zeros(n, dtype=bool)
    if train_indices:
        train_mask[train_indices] = True
    if test_indices:
        test_mask[test_indices] = True
    if nep_train_indices:
        nep_train_mask[nep_train_indices] = True
    unselected = ~(train_mask | test_mask)

    ax.scatter(
        coords[unselected, 0],
        coords[unselected, 1],
        s=4,
        alpha=0.2,
        c="gray",
        label=f"Unselected ({unselected.sum()})",
    )
    if train_mask.any():
        ax.scatter(
            coords[train_mask, 0],
            coords[train_mask, 1],
            s=10,
            alpha=0.4,
            c="tab:red",
            label=f"FPS-selected train ({train_mask.sum()})",
        )
    if test_mask.any():
        ax.scatter(
            coords[test_mask, 0],
            coords[test_mask, 1],
            s=10,
            alpha=0.7,
            c="tab:blue",
            label=f"FPS-selected test ({test_mask.sum()})",
        )
    if nep_train_mask.any():
        ax.scatter(
            coords[nep_train_mask, 0],
            coords[nep_train_mask, 1],
            s=18,
            alpha=0.9,
            c="tab:green",
            label=f"NEP training data ({nep_train_mask.sum()})",
        )

    ax.set_title(title)
    ax.legend()


def _plot(
    descriptors: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
    nep_train_indices: list[int],
    output_path: Path,
    potential_name: str,
) -> None:
    """PCA 2D scatter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca = _fit_pca(descriptors)
    coords = _project_with_pca(descriptors, pca)

    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_panel(
        ax,
        coords,
        train_indices,
        test_indices,
        nep_train_indices,
        f"NEP Descriptor Space - Trained potential: {potential_name}",
    )

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved plot to {output_path}")


def _plot_shared(
    reference_descriptors: np.ndarray,
    other_descriptors: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
    nep_train_indices: list[int],
    output_path: Path,
    reference_name: str,
    other_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca = _fit_pca(reference_descriptors)
    coords_reference = _project_with_pca(reference_descriptors, pca)
    coords_other = _project_with_pca(other_descriptors, pca)
    var = pca.explained_variance_ratio_

    stacked = np.vstack([coords_reference, coords_other])
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    pad = 0.05 * span

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    _plot_panel(
        axes[0],
        coords_reference,
        train_indices,
        test_indices,
        nep_train_indices,
        f"Reference: {reference_name}",
    )
    _plot_panel(
        axes[1],
        coords_other,
        train_indices,
        test_indices,
        nep_train_indices,
        f"Potential: {other_name}",
    )
    for ax in axes:
        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
        ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")

    fig.suptitle("NEP Descriptor Space - PCA Anchored to Reference", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
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
    parser.add_argument("--compare-potential", default=None,
                        help="Second potential folder inside nep/potentials/ for shared-PCA comparison")
    parser.add_argument("--shared-pca", action="store_true",
                        help="Fit one PCA basis on both potentials and plot them side by side")
    args = parser.parse_args()

    project_dir = args.project
    if not project_dir.exists():
        logger.error(f"Project directory not found: {project_dir}")
        sys.exit(1)
    logger.info(f"Project directory: {project_dir}")

    potentials_dir = project_dir / "nep" / "potentials"
    compare_requested = args.compare_potential is not None
    if args.shared_pca and not compare_requested:
        logger.error("--shared-pca requires --compare-potential")
        sys.exit(1)

    try:
        potential_path = _resolve_potential(potentials_dir, args.potential)
        logger.info(f"Using potential: {potential_path.name}")
        compare_potential_path = None
        compare_cache_dir = None
        compare_label = None
        if compare_requested:
            compare_cache_dir, compare_nep_txt, compare_label = _resolve_compare_target(
                project_dir,
                potentials_dir,
                args.compare_potential,
            )
            compare_potential_path = compare_cache_dir
            logger.info(f"Using comparison target: {compare_nep_txt}")
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    nep_txt = potential_path / "nep.txt"
    if not nep_txt.exists():
        logger.error(f"nep.txt not found in {potential_path} - has training completed?")
        sys.exit(1)
    logger.info(f"Using potential: {nep_txt}")

    if compare_requested and compare_nep_txt is None:
        logger.error("Comparison target could not be resolved")
        sys.exit(1)

    structures_path = args.structures or (project_dir / "structures" / "generated" / "generated_structures.xyz")
    train_xyz = args.train_xyz or (project_dir / "structures" / "selected" / "train.xyz")
    test_xyz = args.test_xyz or (project_dir / "structures" / "selected" / "test.xyz")
    output_path = args.output or (
        project_dir / "reports" / (
            "descriptor_space_shared_pca.png" if compare_requested
            else "descriptor_space_trained.png"
        )
    )
    mean_descriptor = not args.atom

    # ----------------------------------------------------------
    # Step 1: Load structures
    # ----------------------------------------------------------
    logger.info("")
    logger.info("Step 1: Loading generated structures")
    t0 = time.perf_counter()
    structures = Structure.read_multiple(str(structures_path))
    ase_structures = ase_read(str(structures_path), index=":", format="extxyz")
    if not isinstance(ase_structures, list):
        ase_structures = [ase_structures]
    logger.info(f"  Loaded {len(structures)} structures ({time.perf_counter() - t0:.1f}s)")

    if len(structures) == 0:
        logger.error("No structures found")
        sys.exit(1)

    # ----------------------------------------------------------
    # Step 2: Load or compute descriptors
    # ----------------------------------------------------------
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

    compare_descriptors = None
    if compare_requested:
        compare_descriptors = _load_or_compute_descriptors(
            compare_cache_dir,
            compare_nep_txt,
            structures,
            mean_descriptor=mean_descriptor,
            batch_size=args.batch,
            no_cache=args.no_cache,
        )
        if compare_descriptors.shape[1] != descriptors.shape[1]:
            logger.error(
                "Descriptor dimension mismatch between potentials: "
                f"{descriptors.shape[1]} vs {compare_descriptors.shape[1]}"
            )
            sys.exit(1)
        logger.info(f"  Comparison descriptor shape: {compare_descriptors.shape}")

    # ----------------------------------------------------------
    # Step 3: Match train / test / nep-train indices
    # ----------------------------------------------------------
    logger.info("")
    logger.info("Step 3: Matching train/test/NEP-training sets")
    train_indices = _match_indices(ase_structures, train_xyz)
    test_indices = _match_indices(ase_structures, test_xyz)
    nep_train_xyz = potential_path / "train.xyz"
    nep_train_indices = _match_indices(ase_structures, nep_train_xyz)
    if not nep_train_indices:
        logger.warning(
            f"  No NEP training structures matched from {nep_train_xyz}\n"
            f"  (train.xyz is copied here by submit_training_job - "
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
    save_kwargs = dict(
        descriptors=descriptors,
        labels=labels,
        train_indices=np.array(train_indices, dtype=int),
        test_indices=np.array(test_indices, dtype=int),
        nep_train_indices=np.array(nep_train_indices, dtype=int),
    )
    if compare_descriptors is not None:
        save_kwargs["compare_descriptors"] = compare_descriptors
        save_kwargs["primary_potential_name"] = np.array(potential_path.name)
        save_kwargs["compare_potential_name"] = np.array(compare_label)
    np.savez(descriptors_out, **save_kwargs)
    logger.info(f"  Saved to {descriptors_out}")

    # ----------------------------------------------------------
    # Step 5: Plot
    # ----------------------------------------------------------
    logger.info("")
    logger.info("Step 5: Plotting descriptor space")
    if compare_descriptors is not None:
        _plot_shared(
            compare_descriptors,
            descriptors,
            train_indices,
            test_indices,
            nep_train_indices,
            output_path,
            compare_label,
            potential_path.name,
        )
    else:
        _plot(descriptors, train_indices, test_indices, nep_train_indices, output_path, potential_path.name)

    logger.info("")
    logger.info("Done.")
    logger.info(f"  Descriptors : {descriptors_out}")
    logger.info(f"  Plot        : {output_path}")


if __name__ == "__main__":
    main()
