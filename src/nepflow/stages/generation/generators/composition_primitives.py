"""Focused composition assignment, quota, and supercell primitives."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from ase import Atoms


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
