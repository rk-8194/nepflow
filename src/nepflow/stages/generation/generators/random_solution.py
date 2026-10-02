"""Random solid-solution generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .composition_primitives import (
    allocate_crystal_quota,
    assign_composition,
    build_target_supercell,
    composition_label,
)


class RandomSolidSolutionGenerator:
    """Create random solid solutions by deterministic injected substitution."""

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
        majority_element = max(composition, key=composition.get)
        for crystal_structure, crystal_quota in quota_plan:
            if crystal_quota == 0:
                continue
            supercell = build_target_supercell(
                majority_element,
                crystal_structure,
                target_n_atoms,
            )
            if supercell is None:
                raise RuntimeError(
                    f"Random solid-solution generation failed for crystal {crystal_structure}"
                )
            for index in range(crystal_quota):
                assigned = assign_composition(supercell, composition, self.rng)
                assigned.info.update(
                    {
                        "composition": dict(composition),
                        "crystal_structure": crystal_structure,
                        "configurational_type": "random_solid_solution",
                        "source": (
                            f"rss-{composition_label(composition)}-"
                            f"{crystal_structure}-{index}"
                        ),
                        "random_seed": self.random_seed,
                    }
                )
                results.append(assigned)
        return results
