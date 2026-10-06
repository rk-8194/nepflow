"""Canonical deterministic target-supercell construction."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping

import numpy as np
from ase import Atoms
from ase.build import bulk

from .generators.composition_primitives import calculate_composition_realization

logger = logging.getLogger(__name__)


class CompositionRealizabilityError(ValueError):
    """Raised when no permitted diagonal repeat satisfies composition tolerance."""


def build_target_supercell(
    source: Atoms | str,
    crystal_structure: str | int | None = None,
    target_n_atoms: int | None = None,
    *,
    composition: Mapping[str, float] | None = None,
    composition_tolerance: float | None = 0.05,
    min_n_atoms: int | None = None,
    max_n_atoms: int | None = None,
    raise_on_error: bool = False,
) -> Atoms | None:
    """Build a target supercell using a deterministic diagonal repeat search.

    ``target_n_atoms`` is a preferred size.  When a composition is supplied,
    only candidate sizes whose canonical integer realization is within
    ``composition_tolerance`` are accepted.  The returned structure remains a
    parent lattice for configurational generators; composition assignment is
    deliberately owned by their shared composition primitive.
    """

    target = target_n_atoms
    if isinstance(source, Atoms):
        if target is None and isinstance(crystal_structure, int):
            target = crystal_structure
        if target is None:
            raise TypeError("target_n_atoms is required for an Atoms source")
        return _expand_atoms(
            source,
            target,
            composition=composition,
            composition_tolerance=composition_tolerance,
            min_n_atoms=min_n_atoms,
            max_n_atoms=max_n_atoms,
            raise_on_error=raise_on_error,
        )

    if not isinstance(crystal_structure, str) or target is None:
        raise TypeError("crystal_structure and target_n_atoms are required for an element source")
    return _build_lattice(
        source,
        crystal_structure,
        target,
        composition=composition,
        composition_tolerance=composition_tolerance,
        min_n_atoms=min_n_atoms,
        max_n_atoms=max_n_atoms,
        raise_on_error=raise_on_error,
    )


def _expand_atoms(
    atoms: Atoms,
    target_n_atoms: int,
    *,
    composition: Mapping[str, float] | None,
    composition_tolerance: float | None,
    min_n_atoms: int | None,
    max_n_atoms: int | None,
    raise_on_error: bool,
) -> Atoms:
    """Expand an existing structure through the same repeat planner."""

    _validate_search_inputs(
        target_n_atoms,
        composition=composition,
        composition_tolerance=composition_tolerance,
        min_n_atoms=min_n_atoms,
        max_n_atoms=max_n_atoms,
    )
    n_atoms = len(atoms)
    if n_atoms == 0:
        return atoms.copy()
    if n_atoms >= target_n_atoms and composition is None:
        if max_n_atoms is not None and n_atoms > max_n_atoms:
            raise CompositionRealizabilityError(
                "existing parent exceeds the configured maximum atom count: "
                f"atom_count={n_atoms}, max_n_atoms={max_n_atoms}"
            )
        if min_n_atoms is None or n_atoms >= min_n_atoms:
            return atoms.copy()
    if n_atoms >= target_n_atoms and composition is not None:
        realization = calculate_composition_realization(composition, n_atoms)
        assert composition_tolerance is not None
        if realization.max_error > composition_tolerance + 1.0e-12:
            repeat = _select_repeat(
                atoms,
                target_n_atoms,
                composition=composition,
                composition_tolerance=composition_tolerance,
                min_n_atoms=min_n_atoms,
                max_n_atoms=max_n_atoms,
            )
            if repeat is None:
                raise CompositionRealizabilityError(
                    "existing parent atom count cannot realize composition within tolerance: "
                    f"atom_count={n_atoms}, error={realization.max_error:.6g}, "
                    f"tolerance={composition_tolerance:.6g}"
                )
            expanded = _repeat(atoms, repeat)
            _record_plan_metadata(
                expanded,
                target_n_atoms=target_n_atoms,
                repeat=repeat,
                composition=composition,
                composition_tolerance=composition_tolerance,
            )
            return expanded
        expanded = atoms.copy()
        _record_plan_metadata(
            expanded,
            target_n_atoms=target_n_atoms,
            repeat=(1, 1, 1),
            composition=composition,
            composition_tolerance=composition_tolerance,
        )
        return expanded

    repeat = _select_repeat(
        atoms,
        target_n_atoms,
        composition=composition,
        composition_tolerance=composition_tolerance,
        min_n_atoms=min_n_atoms,
        max_n_atoms=max_n_atoms,
    )
    if repeat is None:
        if raise_on_error or min_n_atoms is not None or max_n_atoms is not None:
            raise CompositionRealizabilityError(
                "no allowed diagonal repeat satisfies the requested supercell constraints"
            )
        return atoms.copy()
    expanded = _repeat(atoms, repeat)
    _record_plan_metadata(
        expanded,
        target_n_atoms=target_n_atoms,
        repeat=repeat,
        composition=composition,
        composition_tolerance=composition_tolerance,
    )
    return expanded


def _build_lattice(
    element: str,
    crystal_structure: str,
    target_n_atoms: int,
    *,
    composition: Mapping[str, float] | None,
    composition_tolerance: float | None,
    min_n_atoms: int | None,
    max_n_atoms: int | None,
    raise_on_error: bool,
) -> Atoms | None:
    try:
        if crystal_structure == "hcp":
            base = bulk(element, "hcp", a=3.0, c=3.0 * 1.633)
        else:
            base = bulk(element, crystal_structure, a=3.0)
    except Exception:
        # ASE bulk builders expose several structure-specific exception types.
        # ``None`` remains the optional-generator result for ordinary lattice
        # construction failures when fail-fast behaviour was not requested.
        if raise_on_error:
            raise
        logger.debug("Cannot build %s-%s", element, crystal_structure, exc_info=True)
        return None

    _validate_search_inputs(
        target_n_atoms,
        composition=composition,
        composition_tolerance=composition_tolerance,
        min_n_atoms=min_n_atoms,
        max_n_atoms=max_n_atoms,
    )
    if len(base) == 0:
        return None
    repeat = _select_repeat(
        base,
        target_n_atoms,
        composition=composition,
        composition_tolerance=composition_tolerance,
        min_n_atoms=min_n_atoms,
        max_n_atoms=max_n_atoms,
    )
    if repeat is None:
        error = CompositionRealizabilityError(
            "no allowed diagonal repeat satisfies the requested supercell constraints: "
            f"target_n_atoms={target_n_atoms}, base_n_atoms={len(base)}, "
            f"composition={dict(composition) if composition is not None else None}, "
            f"tolerance={composition_tolerance}"
        )
        if (
            composition is not None
            or raise_on_error
            or min_n_atoms is not None
            or max_n_atoms is not None
        ):
            raise error
        return None
    expanded = _repeat(base, repeat)
    _record_plan_metadata(
        expanded,
        target_n_atoms=target_n_atoms,
        repeat=repeat,
        composition=composition,
        composition_tolerance=composition_tolerance,
    )
    return expanded


def _select_repeat(
    parent: Atoms,
    target_n_atoms: int,
    *,
    composition: Mapping[str, float] | None,
    composition_tolerance: float | None,
    min_n_atoms: int | None,
    max_n_atoms: int | None,
) -> tuple[int, int, int] | None:
    """Select the best repeat with stable scientific tie-breaking."""

    n_base = len(parent)
    preferred_scale = max(target_n_atoms / n_base, 1.0)
    max_allowed = max_n_atoms if max_n_atoms is not None else target_n_atoms * 2 + n_base * 8
    search_scale = max(preferred_scale, max_allowed / n_base)
    max_axis = max(2, int(np.ceil(search_scale ** (1.0 / 3.0))) + 3)
    lengths = np.linalg.norm(np.asarray(parent.cell.array, dtype=float), axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        lengths = np.ones(3, dtype=float)

    repeat_candidates = set(itertools.product(range(1, max_axis + 1), repeat=3))
    if composition is not None:
        # A balanced cube-root neighbourhood does not contain factorizations
        # such as 129 = 3 * 3 * 43.  Add factorizations for a bounded window
        # around the preferred atom count so nearby composition-compatible
        # sizes are considered without scanning an unnecessarily large cube.
        window = max(16, int(np.ceil(target_n_atoms * 0.1)))
        lower = max(1, target_n_atoms - window)
        upper = min(max_allowed, target_n_atoms + window)
        for atom_count in range(lower, upper + 1):
            if atom_count % n_base:
                continue
            repeat_candidates.update(_factor_repeat_triples(atom_count // n_base))

    candidates: list[tuple[tuple[float, ...], tuple[int, int, int]]] = []
    for raw_repeat in sorted(repeat_candidates):
        repeat = (int(raw_repeat[0]), int(raw_repeat[1]), int(raw_repeat[2]))
        atom_count = n_base * int(np.prod(repeat))
        if composition is None and atom_count < target_n_atoms:
            continue
        if min_n_atoms is not None and atom_count < min_n_atoms:
            continue
        if max_n_atoms is not None and atom_count > max_n_atoms:
            continue

        realization = None
        if composition is not None:
            assert composition_tolerance is not None
            realization = calculate_composition_realization(composition, atom_count)
            if realization.max_error > composition_tolerance + 1.0e-12:
                continue

        expanded_lengths = lengths * np.asarray(repeat, dtype=float)
        aspect_penalty = float(np.max(expanded_lengths) / np.min(expanded_lengths) - 1.0)
        if composition is None:
            # For unconstrained parent expansion, preserve a balanced periodic
            # cell before considering atom-count distance.  This retains the
            # historical cubic result while allowing anisotropic parents to
            # select a geometry-aware diagonal repeat.
            score = (
                round(aspect_penalty, 12),
                float(abs(atom_count - target_n_atoms)),
                float(atom_count),
                *(float(value) for value in repeat),
            )
        else:
            assert realization is not None
            score = (
                float(abs(atom_count - target_n_atoms)),
                round(aspect_penalty, 12),
                float(realization.max_error),
                float(atom_count),
                *(float(value) for value in repeat),
            )
        candidates.append((score, repeat))

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate[0])
    return candidates[0][1]


def _factor_repeat_triples(product: int) -> set[tuple[int, int, int]]:
    """Return all ordered positive triples with the requested product."""

    triples: set[tuple[int, int, int]] = set()
    for first in range(1, int(np.sqrt(product)) + 1):
        if product % first:
            continue
        quotient = product // first
        for second in range(1, int(np.sqrt(quotient)) + 1):
            if quotient % second:
                continue
            third = quotient // second
            for permutation in itertools.permutations((first, second, third)):
                triples.add((permutation[0], permutation[1], permutation[2]))
    return triples


def _repeat(atoms: Atoms, repeat: tuple[int, int, int]) -> Atoms:
    expanded = atoms.repeat(repeat)
    # ``Atoms.repeat`` preserves these fields, but keeping the assertion local
    # makes the owner explicit for future changes to repeat construction.
    expanded.pbc = atoms.pbc
    return expanded


def _record_plan_metadata(
    atoms: Atoms,
    *,
    target_n_atoms: int,
    repeat: tuple[int, int, int],
    composition: Mapping[str, float] | None,
    composition_tolerance: float | None,
) -> None:
    atoms.info["supercell_repeat"] = repeat
    atoms.info["supercell_target_n_atoms"] = target_n_atoms
    atoms.info["supercell_realized_n_atoms"] = len(atoms)
    if composition is not None:
        realization = calculate_composition_realization(composition, len(atoms))
        atoms.info["composition_target_counts"] = dict(realization.counts)
        atoms.info["composition_target_fractions"] = dict(realization.realized)
        atoms.info["composition_target_error"] = realization.max_error
        if composition_tolerance is not None:
            atoms.info["composition_tolerance"] = float(composition_tolerance)


def _validate_search_inputs(
    target_n_atoms: int,
    *,
    composition: Mapping[str, float] | None,
    composition_tolerance: float | None,
    min_n_atoms: int | None,
    max_n_atoms: int | None,
) -> None:
    if target_n_atoms <= 0:
        raise ValueError("target_n_atoms must be positive")
    if min_n_atoms is not None and min_n_atoms <= 0:
        raise ValueError("min_n_atoms must be positive")
    if max_n_atoms is not None and max_n_atoms <= 0:
        raise ValueError("max_n_atoms must be positive")
    if min_n_atoms is not None and max_n_atoms is not None and min_n_atoms > max_n_atoms:
        raise ValueError("min_n_atoms must not exceed max_n_atoms")
    if composition is not None:
        if composition_tolerance is None:
            raise ValueError("composition_tolerance is required with composition")
        if not np.isfinite(composition_tolerance) or not 0.0 <= composition_tolerance <= 1.0:
            raise ValueError("composition_tolerance must be in [0, 1]")


__all__ = ["CompositionRealizabilityError", "build_target_supercell"]
