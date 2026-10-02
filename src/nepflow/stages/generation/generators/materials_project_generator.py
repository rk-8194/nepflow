"""Materials Project-backed configurational generator."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .materials_project.fetcher import MaterialsProjectFetcher


class MaterialsProjectGenerator:
    """Return known MP phases while retaining requested/realized chemistry."""

    def __init__(
        self,
        fetcher: MaterialsProjectFetcher,
        max_per_composition: int = 5,
        gas_elements: Sequence[str] | None = None,
    ) -> None:
        self.fetcher = fetcher
        self.max_per_composition = max_per_composition
        self.gas_elements = list(gas_elements or [])

    def generate(
        self,
        composition: Mapping[str, float],
        crystal_structures: Sequence[str],
        target_n_atoms: int = 250,
    ) -> list[Any]:
        del target_n_atoms
        elements = [element for element, fraction in composition.items() if fraction > 0]
        if len(elements) == 1:
            atoms_list = self.fetcher.fetch_pure_element_structures(
                elements,
                crystal_structures,
                use_cache=True,
            )
        else:
            atoms_list = self.fetcher.fetch_compounds(
                elements,
                max_per_query=self.max_per_composition,
                use_cache=True,
            )
        for atoms in atoms_list:
            atoms.info["composition"] = dict(composition)
            symbols = atoms.get_chemical_symbols()
            counts = Counter(symbols)
            total = len(symbols)
            atoms.info["actual_composition"] = {
                element: counts[element] / total for element in sorted(counts)
            }
            atoms.info.setdefault("configurational_type", "mp_phase")
        return atoms_list[: self.max_per_composition]

    def generate_gas_phases(
        self,
        metal_elements: Sequence[str],
        gas_elements: Sequence[str],
        target_n_atoms: int = 250,
    ) -> list[Any]:
        del target_n_atoms
        if not gas_elements:
            return []
        all_elements = list(metal_elements) + [
            element for element in gas_elements if element not in metal_elements
        ]
        atoms_list = self.fetcher.fetch_compounds(
            all_elements,
            max_per_query=self.max_per_composition * 2,
            use_cache=True,
        )
        metal_set = set(metal_elements)
        gas_set = set(gas_elements)
        filtered = []
        for atoms in atoms_list:
            symbols = set(atoms.get_chemical_symbols())
            if symbols & gas_set and symbols & metal_set:
                atoms.info.setdefault("configurational_type", "mp_gas_phase")
                atoms.info.setdefault("elements", list(metal_elements))
                atoms.info.setdefault("gas_elements", list(gas_elements))
                filtered.append(atoms)
        return filtered[: self.max_per_composition * 2]
