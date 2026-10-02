"""Focused composition assignment, quota, and supercell primitives."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import logging

import numpy as np
from ase import Atoms
from ase.build import bulk


logger = logging.getLogger("nepflow.generation.composition_primitives")


def allocate_crystal_quota(
    total: int,
    crystal_structures: Sequence[str],
) -> list[tuple[str, int]]:
    """Allocate one per-composition quota across crystals in configured order."""

    if total < 0:
        raise ValueError("n_structures must be non-negative")
    if len(set(crystal_structures)) != len(crystal_structures):
        raise ValueError("crystal_structures must not contain duplicate names")
    if total > 0 and not crystal_structures:
        raise ValueError("positive n_structures requires at least one crystal structure")
    if not crystal_structures:
        return []

    base, remainder = divmod(total, len(crystal_structures))
    return [
        (crystal, base + (1 if index < remainder else 0))
        for index, crystal in enumerate(crystal_structures)
    ]


def build_target_supercell(
    element: str,
    crystal_structure: str,
    target_n_atoms: int,
    *,
    raise_on_error: bool = False,
) -> Atoms | None:
    """Build the accepted Phase 2 isotropic target supercell."""

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
    rep = max(1, round((target_n_atoms / n_base) ** (1.0 / 3.0)))
    return base.repeat(rep)


def assign_composition(
    atoms: Atoms,
    composition: Mapping[str, float],
    rng: np.random.Generator,
) -> Atoms:
    """Assign the requested composition with the Phase 2 largest-remainder rule."""

    n = len(atoms)
    elements = sorted(composition)
    raw = {element: composition[element] * n for element in elements}
    counts = {element: int(np.floor(raw[element])) for element in elements}
    remainder = n - sum(counts.values())
    fractional_parts = sorted(
        elements,
        key=lambda element: raw[element] - counts[element],
        reverse=True,
    )
    for index in range(remainder):
        counts[fractional_parts[index % len(fractional_parts)]] += 1

    symbols: list[str] = []
    for element in elements:
        symbols.extend([element] * counts[element])
    rng.shuffle(symbols)

    result = atoms.copy()
    result.set_chemical_symbols(symbols[:n])
    result.info["actual_composition"] = {
        element: symbols[:n].count(element) / n for element in elements
    }
    return result


def realized_composition(atoms: Atoms) -> dict[str, float]:
    """Return the measured composition of an ASE structure."""

    symbols = atoms.get_chemical_symbols()
    counts = Counter(symbols)
    total = len(symbols)
    if total == 0:
        return {}
    return {element: counts[element] / total for element in sorted(counts)}


def composition_label(composition: Mapping[str, float]) -> str:
    """Return the stable Phase 2 source-label spelling."""

    parts: list[str] = []
    for element in sorted(composition, key=lambda item: -composition[item]):
        fraction = composition[element]
        parts.append(element if fraction >= 1.0 else f"{element}{fraction:.2g}")
    return "-".join(parts)
