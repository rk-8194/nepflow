"""Segregated/phase-separated configurational generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from nepflow.stages.generation.supercell import build_target_supercell

from .composition_primitives import (
    allocate_crystal_quota,
    composition_label,
)


class SegregatedGenerator:
    """Generate layer-segregated structures with deterministic provenance."""

    def __init__(
        self,
        n_structures: int = 3,
        random_seed: int = 42,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_structures = n_structures
        self.random_seed = random_seed
        self.rng = rng if rng is not None else np.random.default_rng(random_seed)

    def generate(
        self,
        composition: Mapping[str, float],
        crystal_structures: Sequence[str],
        target_n_atoms: int = 250,
    ) -> list[Any]:
        quota_plan = allocate_crystal_quota(self.n_structures, crystal_structures)
        if sum(1 for fraction in composition.values() if fraction > 0) <= 1:
            return []

        results: list[Any] = []
        majority_element = max(composition, key=lambda element: composition[element])
        active_elements = sorted(
            element for element, fraction in composition.items() if fraction > 0
        )
        strategies = ["x", "y", "z"]

        for crystal_structure, crystal_quota in quota_plan:
            if crystal_quota == 0:
                continue
            if crystal_quota > len(strategies):
                raise ValueError(
                    f"Segregated generation supports at most {len(strategies)} "
                    f"structures per crystal, requested {crystal_quota} for {crystal_structure}"
                )
            supercell = build_target_supercell(
                majority_element,
                crystal_structure,
                target_n_atoms,
            )
            if supercell is None:
                raise RuntimeError(f"Segregated generation failed for crystal {crystal_structure}")
            for strategy in strategies[:crystal_quota]:
                segregated = self._layered_segregation(
                    supercell,
                    composition,
                    active_elements,
                    strategy,
                )
                segregated.info.update(
                    {
                        "composition": dict(composition),
                        "crystal_structure": crystal_structure,
                        "configurational_type": "segregated",
                        "segregation_axis": strategy,
                        "source": (
                            f"seg-{composition_label(composition)}-{crystal_structure}-{strategy}"
                        ),
                        "random_seed": self.random_seed,
                    }
                )
                results.append(segregated)
        return results

    def _layered_segregation(
        self,
        supercell: Any,
        composition: Mapping[str, float],
        active_elements: Sequence[str],
        axis: str,
    ) -> Any:
        atoms = supercell.copy()
        n = len(atoms)
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        positions = atoms.get_positions()
        sorted_indices = np.argsort(positions[:, axis_index])

        symbols = [""] * n
        offset = 0
        for element in active_elements:
            count = min(round(composition[element] * n), n - offset)
            for index in range(count):
                symbols[sorted_indices[offset + index]] = element
            offset += count
        for index in range(offset, n):
            symbols[sorted_indices[index]] = active_elements[-1]

        atoms.set_chemical_symbols(symbols)
        atoms.info["actual_composition"] = {
            element: symbols.count(element) / n for element in active_elements
        }
        return atoms
