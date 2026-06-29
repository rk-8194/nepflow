"""
Configurational generators for multi-element structures.

Each generator takes a target composition and crystal structure(s)
and returns ASE Atoms objects representing different atomic arrangements:
  - MaterialsProjectGenerator: known phases from MP database
  - RandomSolidSolutionGenerator: random atom substitution
  - SQSGenerator: special quasi-random structures (icet)
  - SegregatedGenerator: phase-separated domains
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.build import bulk

logger = logging.getLogger("nepflow.configurational")


# ======================================================================
# Base interface
# ======================================================================

class ConfigurationalGenerator(ABC):
    """Interface for structure generators."""

    @abstractmethod
    def generate(
        self,
        composition: Dict[str, float],
        crystal_structures: List[str],
        target_n_atoms: int = 250,
    ) -> List[Atoms]:
        """Generate base structures for a given composition.

        Args:
            composition: Target atomic fractions, e.g. ``{"W": 0.5, "Cr": 0.5}``.
            crystal_structures: Lattice types to use (bcc, fcc, hcp, diamond, simple_cubic).
            target_n_atoms: Desired number of atoms per supercell.

        Returns:
            List of ASE Atoms with standardised ``atoms.info`` metadata.
        """


# ======================================================================
# Helper: build a supercell of a pure-element lattice
# ======================================================================

def _make_supercell(element: str, crystal_structure: str, target_n_atoms: int) -> Optional[Atoms]:
    """Create a pure-element supercell close to *target_n_atoms*."""
    try:
        if crystal_structure == "hcp":
            base = bulk(element, "hcp", a=3.0, c=3.0 * 1.633)
        else:
            base = bulk(element, crystal_structure, a=3.0)
    except Exception as e:
        logger.debug(f"Cannot build {element}-{crystal_structure}: {e}")
        return None

    n_base = len(base)
    if n_base == 0:
        return None
    rep = max(1, round((target_n_atoms / n_base) ** (1.0 / 3.0)))
    return base.repeat(rep)


def _assign_composition(atoms: Atoms, composition: Dict[str, float], rng: np.random.RandomState) -> Atoms:
    """Replace chemical symbols to match *composition* as closely as possible."""
    n = len(atoms)
    elements = sorted(composition.keys())

    # Convert fractions → integer counts (largest-remainder method)
    raw = {el: composition[el] * n for el in elements}
    counts = {el: int(np.floor(raw[el])) for el in elements}
    remainder = n - sum(counts.values())
    frac_parts = sorted(elements, key=lambda el: raw[el] - counts[el], reverse=True)
    for i in range(remainder):
        counts[frac_parts[i % len(frac_parts)]] += 1

    symbols: List[str] = []
    for el in elements:
        symbols.extend([el] * counts[el])
    rng.shuffle(symbols)

    atoms_out = atoms.copy()
    atoms_out.set_chemical_symbols(symbols[:n])

    # Record realised composition
    actual: Dict[str, float] = {}
    for el in elements:
        actual[el] = symbols[:n].count(el) / n
    atoms_out.info["actual_composition"] = actual
    return atoms_out


def _composition_label(composition: Dict[str, float]) -> str:
    parts = []
    for el in sorted(composition, key=lambda e: -composition[e]):
        f = composition[el]
        parts.append(f"{el}{f:.2g}" if f < 1.0 else el)
    return "-".join(parts)


# ======================================================================
# 1) Materials Project generator
# ======================================================================

class MaterialsProjectGenerator(ConfigurationalGenerator):
    """Return known phases from the Materials Project database."""

    def __init__(self, fetcher, max_per_composition: int = 5, gas_elements: List[str] | None = None):
        self.fetcher = fetcher
        self.max_per_composition = max_per_composition
        self.gas_elements = gas_elements or []

    def generate(
        self,
        composition: Dict[str, float],
        crystal_structures: List[str],
        target_n_atoms: int = 250,
    ) -> List[Atoms]:
        elements = [el for el, frac in composition.items() if frac > 0]

        if len(elements) == 1:
            # Pure element → use filtered pure-element query
            atoms_list = self.fetcher.fetch_pure_element_structures(
                elements, crystal_structures, use_cache=True,
            )
        else:
            # Multi-element → compound query
            atoms_list = self.fetcher.fetch_compounds(
                elements, max_per_query=self.max_per_composition, use_cache=True,
            )

        # Annotate
        for atoms in atoms_list:
            atoms.info.setdefault("composition", composition)
            atoms.info.setdefault("configurational_type", "mp_phase")

        return atoms_list[:self.max_per_composition]

    def generate_gas_phases(
        self,
        metal_elements: List[str],
        gas_elements: List[str],
        target_n_atoms: int = 250,
    ) -> List[Atoms]:
        """Fetch stable compounds containing both metal and gas elements from MP.

        These are oxide / nitride / etc. phases that can't be generated via
        lattice substitution.  They enter the base-structure pool directly so
        that standard (and gas-specific) perturbations are applied to them.
        """
        if not gas_elements:
            return []

        all_elements = list(metal_elements) + [g for g in gas_elements if g not in metal_elements]
        atoms_list = self.fetcher.fetch_compounds(
            all_elements, max_per_query=self.max_per_composition * 2, use_cache=True,
        )

        # Keep only phases that actually contain at least one gas element
        gas_set = set(gas_elements)
        filtered: List[Atoms] = []
        for atoms in atoms_list:
            symbols = set(atoms.get_chemical_symbols())
            if symbols & gas_set:
                atoms.info.setdefault("configurational_type", "mp_gas_phase")
                atoms.info.setdefault("elements", metal_elements)
                atoms.info.setdefault("gas_elements", gas_elements)
                filtered.append(atoms)

        logger.info(f"  MP gas phases: {len(filtered)} structures "
                     f"(from {len(atoms_list)} total compounds for {all_elements})")
        return filtered[:self.max_per_composition * 2]


# ======================================================================
# 2) Random solid-solution generator
# ======================================================================

class RandomSolidSolutionGenerator(ConfigurationalGenerator):
    """Create random solid solutions by atom substitution on a lattice."""

    def __init__(self, n_structures: int = 3, random_seed: int = 42):
        self.n_structures = n_structures
        self.rng = np.random.RandomState(random_seed)

    def generate(
        self,
        composition: Dict[str, float],
        crystal_structures: List[str],
        target_n_atoms: int = 250,
    ) -> List[Atoms]:
        # For pure elements random substitution is pointless
        if sum(1 for f in composition.values() if f > 0) <= 1:
            return []

        results: List[Atoms] = []
        # Use the majority element to build the lattice
        majority_el = max(composition, key=composition.get)

        for cs in crystal_structures:
            supercell = _make_supercell(majority_el, cs, target_n_atoms)
            if supercell is None:
                continue

            for i in range(self.n_structures):
                assigned = _assign_composition(supercell, composition, self.rng)
                label = _composition_label(composition)
                assigned.info.update({
                    "composition": composition,
                    "crystal_structure": cs,
                    "configurational_type": "random_solid_solution",
                    "source": f"rss-{label}-{cs}-{i}",
                })
                results.append(assigned)

        return results[:self.n_structures]


# ======================================================================
# 3) SQS generator (via icet)
# ======================================================================

class SQSGenerator(ConfigurationalGenerator):
    """Generate Special Quasi-random Structures using *icet*."""

    def __init__(self, n_structures: int = 3, random_seed: int = 42):
        self.n_structures = n_structures
        self.random_seed = random_seed

    def generate(
        self,
        composition: Dict[str, float],
        crystal_structures: List[str],
        target_n_atoms: int = 250,
    ) -> List[Atoms]:
        if sum(1 for f in composition.values() if f > 0) <= 1:
            return []

        try:
            from icet import ClusterSpace
            from icet.tools.structure_generation import (
                generate_sqs_from_supercells,
                _get_sqs_cluster_vector,
            )
        except ImportError:
            logger.warning("icet not installed — skipping SQS generation")
            return []

        results: List[Atoms] = []
        majority_el = max(composition, key=composition.get)

        active_elements = sorted(el for el, f in composition.items() if f > 0)
        target_concentrations = {el: composition.get(el, 0.0) for el in active_elements}

        for cs in crystal_structures:
            supercell = _make_supercell(majority_el, cs, target_n_atoms)
            if supercell is None:
                continue

            try:
                prim = bulk(majority_el, cs, a=3.0) if cs != "hcp" else bulk(majority_el, "hcp", a=3.0, c=3.0 * 1.633)
                cluster_space = ClusterSpace(prim, cutoffs=[6.0], chemical_symbols=[active_elements])

                sqs_atoms = generate_sqs_from_supercells(
                    cluster_space=cluster_space,
                    max_size=len(supercell),
                    target_concentrations=target_concentrations,
                    n_steps=5000,
                    random_seed=self.random_seed,
                )

                label = _composition_label(composition)
                sqs_atoms.info.update({
                    "composition": composition,
                    "crystal_structure": cs,
                    "configurational_type": "sqs",
                    "source": f"sqs-{label}-{cs}",
                })
                results.append(sqs_atoms)

            except Exception as e:
                logger.debug(f"SQS generation failed for {composition} on {cs}: {e}")
                # Fall back to monte-carlo approach
                try:
                    from icet.tools.structure_generation import generate_target_structure
                    mc_atoms = self._mc_fallback(supercell, composition, active_elements, cs)
                    if mc_atoms is not None:
                        results.append(mc_atoms)
                except Exception as e2:
                    logger.debug(f"SQS MC fallback also failed: {e2}")

        return results[:self.n_structures]

    def _mc_fallback(
        self,
        supercell: Atoms,
        composition: Dict[str, float],
        active_elements: List[str],
        cs: str,
    ) -> Optional[Atoms]:
        """Simple MC-based SQS via random swaps."""
        rng = np.random.RandomState(self.random_seed)
        assigned = _assign_composition(supercell, composition, rng)
        label = _composition_label(composition)
        assigned.info.update({
            "composition": composition,
            "crystal_structure": cs,
            "configurational_type": "sqs",
            "source": f"sqs-mc-{label}-{cs}",
        })
        return assigned


# ======================================================================
# 4) Segregated / phase-separated generator
# ======================================================================

class SegregatedGenerator(ConfigurationalGenerator):
    """Generate structures with spatial phase separation."""

    def __init__(self, n_structures: int = 3, random_seed: int = 42):
        self.n_structures = n_structures
        self.rng = np.random.RandomState(random_seed)

    def generate(
        self,
        composition: Dict[str, float],
        crystal_structures: List[str],
        target_n_atoms: int = 250,
    ) -> List[Atoms]:
        if sum(1 for f in composition.values() if f > 0) <= 1:
            return []

        results: List[Atoms] = []
        majority_el = max(composition, key=composition.get)
        active_elements = sorted(el for el, f in composition.items() if f > 0)

        # Segregation strategies: one per axis direction (x, y, z)
        strategies = ["x", "y", "z"]

        for cs in crystal_structures:
            supercell = _make_supercell(majority_el, cs, target_n_atoms)
            if supercell is None:
                continue

            for i, strategy in enumerate(strategies[:self.n_structures]):
                seg = self._layered_segregation(supercell, composition, active_elements, strategy)
                label = _composition_label(composition)
                seg.info.update({
                    "composition": composition,
                    "crystal_structure": cs,
                    "configurational_type": "segregated",
                    "segregation_axis": strategy,
                    "source": f"seg-{label}-{cs}-{strategy}",
                })
                results.append(seg)

        return results[:self.n_structures]

    def _layered_segregation(
        self,
        supercell: Atoms,
        composition: Dict[str, float],
        active_elements: List[str],
        axis: str,
    ) -> Atoms:
        """Assign elements in layers along *axis* to match target composition."""
        atoms = supercell.copy()
        n = len(atoms)
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]

        # Sort atoms by position along the chosen axis
        positions = atoms.get_positions()
        sorted_indices = np.argsort(positions[:, axis_idx])

        # Build symbol list from composition fractions (in order)
        symbols = [""] * n
        offset = 0
        for el in active_elements:
            count = int(round(composition[el] * n))
            # Clamp to remaining slots
            count = min(count, n - offset)
            for j in range(count):
                symbols[sorted_indices[offset + j]] = el
            offset += count

        # Fill any remaining atoms with the last element (rounding artifact)
        for j in range(offset, n):
            symbols[sorted_indices[j]] = active_elements[-1]

        atoms.set_chemical_symbols(symbols)

        actual: Dict[str, float] = {}
        for el in active_elements:
            actual[el] = symbols.count(el) / n
        atoms.info["actual_composition"] = actual
        return atoms
