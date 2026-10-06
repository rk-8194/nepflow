"""Focused composition assignment, quota, and supercell primitives."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass(frozen=True, slots=True)
class CompositionRealization:
    """Deterministic integer realization of a requested composition."""

    requested: dict[str, float]
    counts: dict[str, int]
    realized: dict[str, float]
    max_error: float
    atom_count: int

    @property
    def realized_composition(self) -> dict[str, float]:
        """Return the realized fractions using the common spelling."""

        return dict(self.realized)

    @property
    def realised(self) -> dict[str, float]:
        """Return the realized fractions using the project spelling."""

        return dict(self.realized)

    @property
    def composition_error(self) -> float:
        """Return the configured metric's value."""

        return self.max_error

    @property
    def max_composition_error(self) -> float:
        """Return the configured metric's value with an explicit name."""

        return self.max_error

    @property
    def realized_counts(self) -> dict[str, int]:
        """Return the realized integer counts."""

        return dict(self.counts)

    @property
    def realised_counts(self) -> dict[str, int]:
        """Return the realized integer counts using the project spelling."""

        return dict(self.counts)

    @property
    def realized_fractions(self) -> dict[str, float]:
        """Return the realized fractions."""

        return dict(self.realized)

    @property
    def realised_fractions(self) -> dict[str, float]:
        """Return the realized fractions using the project spelling."""

        return dict(self.realized)


def calculate_composition_realization(
    composition: Mapping[str, float],
    atom_count: int,
) -> CompositionRealization:
    """Calculate a stable largest-remainder integer composition.

    Elements are ordered lexically for every tie.  The function is the single
    owner of integer composition planning; callers should not independently
    round fractions.
    """

    if atom_count <= 0:
        raise ValueError("atom_count must be positive")
    if not composition:
        raise ValueError("composition must contain at least one element")

    requested = {element: float(composition[element]) for element in sorted(composition)}
    if any(not np.isfinite(value) or value < 0.0 for value in requested.values()):
        raise ValueError("composition fractions must be finite and non-negative")
    total = sum(requested.values())
    if not np.isclose(total, 1.0, rtol=1.0e-9, atol=1.0e-9):
        raise ValueError("composition fractions must sum to one")

    raw = {element: fraction * atom_count for element, fraction in requested.items()}
    counts = {element: int(np.floor(value)) for element, value in raw.items()}
    remainder = atom_count - sum(counts.values())
    ranked = sorted(
        requested,
        key=lambda element: (-(raw[element] - counts[element]), element),
    )
    for element in ranked[:remainder]:
        counts[element] += 1

    realized = {element: counts[element] / atom_count for element in requested}
    max_error = max(abs(realized[element] - requested[element]) for element in requested)
    return CompositionRealization(
        requested=requested,
        counts=counts,
        realized=realized,
        max_error=float(max_error),
        atom_count=atom_count,
    )


# The shorter name is useful at call sites and remains an alias to the
# canonical calculation rather than a second implementation.
realize_composition = calculate_composition_realization


def measure_composition_realization(
    atoms: Atoms,
    requested: Mapping[str, float],
) -> CompositionRealization:
    """Measure an existing structure against a requested composition."""

    symbols = atoms.get_chemical_symbols()
    atom_count = len(symbols)
    if atom_count <= 0:
        raise ValueError("cannot measure composition for an empty structure")
    requested_values = {element: float(value) for element, value in requested.items()}
    if any(not np.isfinite(value) or value < 0.0 for value in requested_values.values()):
        raise ValueError("composition fractions must be finite and non-negative")
    if any(element not in requested_values for element in symbols):
        requested_values.update({element: 0.0 for element in sorted(set(symbols))})
    counts = Counter(symbols)
    ordered_elements = sorted(set(requested_values) | set(counts))
    requested_values = {element: requested_values.get(element, 0.0) for element in ordered_elements}
    total = sum(requested_values.values())
    if not np.isclose(total, 1.0, rtol=1.0e-9, atol=1.0e-9):
        raise ValueError("composition fractions must sum to one")
    realized = {element: counts[element] / atom_count for element in ordered_elements}
    max_error = max(
        abs(realized[element] - requested_values[element]) for element in ordered_elements
    )
    return CompositionRealization(
        requested=requested_values,
        counts={element: counts[element] for element in ordered_elements},
        realized=realized,
        max_error=float(max_error),
        atom_count=atom_count,
    )


def record_composition_metadata(
    atoms: Atoms,
    requested: Mapping[str, float],
    *,
    tolerance: float | None = None,
    require_tolerance: bool = False,
) -> CompositionRealization:
    """Record requested/realized composition fields on an ASE structure."""

    realization = measure_composition_realization(atoms, requested)
    if tolerance is not None and (not np.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0):
        raise ValueError("composition tolerance must be in [0, 1]")
    if require_tolerance:
        if tolerance is None:
            raise ValueError("a composition tolerance is required for validation")
        if realization.max_error > tolerance + 1.0e-12:
            raise ValueError(
                "realized composition exceeds tolerance: "
                f"error={realization.max_error:.6g}, tolerance={tolerance:.6g}"
            )

    metadata = atoms.info
    metadata["requested_composition"] = dict(realization.requested)
    metadata["actual_composition"] = dict(realization.realized)
    metadata["realized_composition"] = dict(realization.realized)
    metadata["composition_counts"] = dict(realization.counts)
    metadata["realized_composition_counts"] = dict(realization.counts)
    metadata["composition_error"] = realization.max_error
    metadata["max_composition_error"] = realization.max_error
    if tolerance is not None:
        metadata["composition_tolerance"] = float(tolerance)
    return realization


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
    *,
    tolerance: float | None = None,
    require_tolerance: bool = False,
) -> Atoms:
    """Assign the canonical deterministic composition realization."""

    realization = calculate_composition_realization(composition, len(atoms))

    symbols: list[str] = []
    for element, count in realization.counts.items():
        symbols.extend([element] * count)
    rng.shuffle(symbols)

    result = atoms.copy()
    result.set_chemical_symbols(symbols)
    record_composition_metadata(
        result,
        composition,
        tolerance=tolerance,
        require_tolerance=require_tolerance,
    )
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


__all__ = [
    "CompositionRealization",
    "allocate_crystal_quota",
    "assign_composition",
    "calculate_composition_realization",
    "composition_label",
    "measure_composition_realization",
    "realize_composition",
    "realized_composition",
    "record_composition_metadata",
]
