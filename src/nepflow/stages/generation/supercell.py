"""Canonical target-supercell construction for generation workflows."""

from __future__ import annotations

import logging

from ase import Atoms
from ase.build import bulk

logger = logging.getLogger(__name__)


def build_target_supercell(
    source: Atoms | str,
    crystal_structure: str | int | None = None,
    target_n_atoms: int | None = None,
    *,
    raise_on_error: bool = False,
) -> Atoms | None:
    """Build the accepted target supercell for a lattice or existing structure.

    The lattice form preserves the configurational-generator convention.  The
    ``Atoms`` form preserves the perturbation-engine convention, including
    species, cell, and periodic-boundary semantics.
    """

    if isinstance(source, Atoms):
        target = target_n_atoms
        if target is None and isinstance(crystal_structure, int):
            target = crystal_structure
        if target is None:
            raise TypeError("target_n_atoms is required for an Atoms source")
        return _expand_atoms(source, target)

    if not isinstance(crystal_structure, str) or target_n_atoms is None:
        raise TypeError("crystal_structure and target_n_atoms are required for an element source")
    return _build_lattice(
        source,
        crystal_structure,
        target_n_atoms,
        raise_on_error=raise_on_error,
    )


def _expand_atoms(atoms: Atoms, target_n_atoms: int) -> Atoms:
    """Repeat an existing structure using the accepted isotropic rule."""

    n_atoms = len(atoms)
    if n_atoms == 0 or n_atoms >= target_n_atoms:
        return atoms.copy()
    repetitions = max(1, round((target_n_atoms / n_atoms) ** (1.0 / 3.0)))
    expanded = atoms.repeat(repetitions)
    # ``Atoms.repeat`` preserves cell, PBC, and species; keep this assertion
    # local so future changes cannot silently alter the scientific contract.
    expanded.pbc = atoms.pbc
    return expanded


def _build_lattice(
    element: str,
    crystal_structure: str,
    target_n_atoms: int,
    *,
    raise_on_error: bool,
) -> Atoms | None:
    try:
        if crystal_structure == "hcp":
            base = bulk(element, "hcp", a=3.0, c=3.0 * 1.633)
        else:
            base = bulk(element, crystal_structure, a=3.0)
    except Exception:
        if raise_on_error:
            raise
        logger.debug("Cannot build %s-%s", element, crystal_structure, exc_info=True)
        return None

    n_base = len(base)
    if n_base == 0:
        return None
    repetitions = max(1, round((target_n_atoms / n_base) ** (1.0 / 3.0)))
    return base.repeat(repetitions)


__all__ = ["build_target_supercell"]
