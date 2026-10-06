"""Vacancy, interstitial, and gas-defect perturbation families."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from ase import Atom

from nepflow.domain.identities import calculate_structure_id

from .models import PerturbationSettings, derive_child_seed

Annotate = Callable[..., Any]


def _gas_interstitial_distance(settings: PerturbationSettings) -> float:
    value = settings.gas_interstitial_d_min
    return settings.interstitial_d_min if value is None else value


def vacancies(
    supercell: Any,
    base: Any,
    n: int,
    settings: PerturbationSettings,
    rng: np.random.RandomState | None,
    annotate: Annotate,
    *,
    seed: int | None = None,
) -> list[Any]:
    """Generate vacancy structures with one deterministic RNG per slot.

    ``vacancy_range`` is a fraction of the periodic supercell atom count; at
    least one atom is removed while retaining one atom.  Candidate provenance
    records the realized count and no input structure is mutated.
    """
    del rng
    base_structure_id = calculate_structure_id(base)
    root_seed = settings.random_seed if seed is None else int(seed)
    output: list[Any] = []
    for index in range(n):
        child_seed = derive_child_seed(base_structure_id, root_seed, "vacancy", index)
        slot_rng = np.random.RandomState(child_seed)
        vacancy = supercell.copy()
        fraction = slot_rng.uniform(*settings.vacancy_range)
        n_remove = max(1, int(fraction * len(vacancy)))
        n_remove = min(n_remove, len(vacancy) - 1)
        keep = sorted(slot_rng.choice(len(vacancy), size=len(vacancy) - n_remove, replace=False))
        vacancy = vacancy[keep]
        annotate(
            vacancy,
            base,
            "vacancy",
            random_seed=child_seed,
            parameters={
                "requested_n_vacancies": n_remove,
                "realised_n_vacancies": n_remove,
                "n_vacancies": n_remove,
            },
            operation_id=f"vacancy:{index}",
        )
        output.append(vacancy)
    return output


def interstitials(
    supercell: Any,
    base: Any,
    n: int,
    settings: PerturbationSettings,
    rng: np.random.RandomState | None,
    annotate: Annotate,
    *,
    seed: int | None = None,
) -> list[Any]:
    """Generate host interstitials at least ``d_min`` Angstrom apart.

    Positions are sampled in fractional coordinates and wrapped through the
    supplied periodic cell.  Each slot has its own deterministic random source;
    failures to find a site are recorded by omitting that attempted insertion.
    """
    elements = base.info.get("elements", sorted(set(supercell.get_chemical_symbols())))
    if isinstance(elements, str):
        elements = [elements]
    return _insert_interstitials(
        supercell,
        base,
        n,
        settings,
        rng,
        list(elements),
        settings.interstitial_d_min,
        "interstitial",
        "n_interstitials",
        annotate,
        seed=seed,
    )


def gas_interstitials(
    supercell: Any,
    base: Any,
    n: int,
    settings: PerturbationSettings,
    rng: np.random.RandomState | None,
    annotate: Annotate,
    *,
    seed: int | None = None,
) -> list[Any]:
    """Generate gas-species interstitials with the configured cutoff."""
    if not settings.gas_elements:
        return []
    return _insert_interstitials(
        supercell,
        base,
        n,
        settings,
        rng,
        list(settings.gas_elements),
        _gas_interstitial_distance(settings),
        "gas_interstitial",
        "n_gas_interstitials",
        annotate,
        seed=seed,
    )


def vacancy_interstitial(
    supercell: Any,
    base: Any,
    n: int,
    settings: PerturbationSettings,
    rng: np.random.RandomState | None,
    annotate: Annotate,
    *,
    seed: int | None = None,
) -> list[Any]:
    """Generate combined vacancy/interstitial structures in a periodic cell."""
    all_elements = base.info.get("elements", sorted(set(supercell.get_chemical_symbols())))
    if isinstance(all_elements, str):
        all_elements = [all_elements]
    all_elements = list(all_elements) + [
        element for element in settings.gas_elements if element not in all_elements
    ]
    del rng
    base_structure_id = calculate_structure_id(base)
    root_seed = settings.random_seed if seed is None else int(seed)
    cell = np.array(supercell.cell)
    inv_cell = np.linalg.inv(cell)
    output: list[Any] = []
    for index in range(n):
        child_seed = derive_child_seed(base_structure_id, root_seed, "vacancy_interstitial", index)
        slot_rng = np.random.RandomState(child_seed)
        vacancy = supercell.copy()
        fraction = slot_rng.uniform(*settings.vacancy_range)
        n_remove = max(1, int(fraction * len(vacancy)))
        n_remove = min(n_remove, len(vacancy) - 1)
        keep = sorted(slot_rng.choice(len(vacancy), size=len(vacancy) - n_remove, replace=False))
        vacancy = vacancy[keep]
        positions = vacancy.get_positions().copy()
        new_positions: list[np.ndarray] = []
        new_symbols: list[str] = []
        fraction = slot_rng.uniform(*settings.interstitial_range)
        n_add = max(1, int(fraction * len(supercell)))
        for _ in range(n_add):
            element = all_elements[slot_rng.randint(len(all_elements))]
            d_min = (
                _gas_interstitial_distance(settings)
                if element in settings.gas_elements
                else settings.interstitial_d_min
            )
            position = find_interstitial_site(
                positions,
                cell,
                inv_cell,
                slot_rng,
                max_attempts=500,
                d_min=d_min,
            )
            if position is not None:
                new_positions.append(position)
                new_symbols.append(element)
                positions = np.vstack([positions, position])
        for position, symbol in zip(new_positions, new_symbols):
            vacancy.append(Atom(symbol=symbol, position=position))
        annotate(
            vacancy,
            base,
            "vacancy_interstitial",
            random_seed=child_seed,
            parameters={
                "requested_n_vacancies": n_remove,
                "realised_n_vacancies": n_remove,
                "n_vacancies": n_remove,
                "requested_n_interstitials": n_add,
                "realised_n_interstitials": len(new_positions),
                "n_interstitials": len(new_positions),
            },
            operation_id=f"vacancy-interstitial:{index}",
        )
        output.append(vacancy)
    return output


def gas_in_vacancy(
    supercell: Any,
    base: Any,
    n: int,
    settings: PerturbationSettings,
    rng: np.random.RandomState | None,
    annotate: Annotate,
    *,
    seed: int | None = None,
) -> list[Any]:
    """Place configured gas species around a vacancy within the cell.

    Distances and search radius are in Angstroms; periodic minimum-image
    distances enforce the configured gas cutoff.
    """
    if not settings.gas_elements:
        return []
    del rng
    base_structure_id = calculate_structure_id(base)
    root_seed = settings.random_seed if seed is None else int(seed)
    cell = np.array(supercell.cell)
    inv_cell = np.linalg.inv(cell)
    output: list[Any] = []
    for index in range(n):
        child_seed = derive_child_seed(base_structure_id, root_seed, "gas_in_vacancy", index)
        slot_rng = np.random.RandomState(child_seed)
        vacancy = supercell.copy()
        n_atoms = len(vacancy)
        vacancy_index = slot_rng.randint(n_atoms)
        vacancy_position = vacancy.get_positions()[vacancy_index].copy()
        vacancy_element = vacancy.get_chemical_symbols()[vacancy_index]
        vacancy = vacancy[[item for item in range(n_atoms) if item != vacancy_index]]
        n_gas = slot_rng.randint(1, settings.max_gas_occupancy + 1)
        remaining_positions = vacancy.get_positions().copy()
        placed = 0
        gas_species: list[str] = []
        for gas_index in range(n_gas):
            gas_element = settings.gas_elements[slot_rng.randint(len(settings.gas_elements))]
            if gas_index == 0:
                offset = slot_rng.normal(scale=0.1, size=3)
                candidate = vacancy_position + offset
                fractional = candidate @ inv_cell
                fractional -= np.floor(fractional)
                candidate = fractional @ cell
                if len(remaining_positions) > 0:
                    distance = _minimum_image_distances(
                        remaining_positions, candidate, cell, inv_cell
                    )
                    if float(np.min(distance)) < _gas_interstitial_distance(settings) ** 2:
                        candidate = vacancy_position.copy()
                position = candidate
            else:
                position = find_site_near(
                    vacancy_position,
                    remaining_positions,
                    cell,
                    inv_cell,
                    slot_rng,
                    radius=2.5,
                    d_min=_gas_interstitial_distance(settings),
                    max_attempts=500,
                )
            if position is not None:
                vacancy.append(Atom(symbol=gas_element, position=position))
                remaining_positions = np.vstack([remaining_positions, position])
                placed += 1
                gas_species.append(gas_element)
        annotate(
            vacancy,
            base,
            "gas_in_vacancy",
            random_seed=child_seed,
            parameters={
                "requested_n_vacancies": 1,
                "realised_n_vacancies": 1,
                "n_gas_atoms": placed,
                "requested_n_gas_atoms": n_gas,
                "realised_n_gas_atoms": placed,
                "vacancy_element": vacancy_element,
                "gas_species": gas_species,
            },
            operation_id=f"gas-in-vacancy:{index}",
        )
        output.append(vacancy)
    return output


def find_interstitial_site(
    positions: np.ndarray,
    cell: np.ndarray,
    inv_cell: np.ndarray,
    rng: np.random.RandomState,
    *,
    max_attempts: int = 500,
    d_min: float,
) -> np.ndarray | None:
    """Find a random fractional-cell site at least ``d_min`` Angstrom away.

    ``positions`` has shape ``(n_atoms, 3)`` and ``cell``/``inv_cell`` have
    shape ``(3, 3)``.  ``None`` means no valid site was found in
    ``max_attempts`` trials.
    """
    d_min_squared = d_min**2
    for _ in range(max_attempts):
        fractional = rng.random(3)
        position = fractional @ cell
        distances = _minimum_image_distances(positions, position, cell, inv_cell)
        if len(distances) == 0 or float(np.min(distances)) >= d_min_squared:
            return position
    return None


def find_site_near(
    centre: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    inv_cell: np.ndarray,
    rng: np.random.RandomState,
    *,
    radius: float = 2.0,
    d_min: float,
    max_attempts: int = 500,
) -> np.ndarray | None:
    """Find a wrapped site near ``centre`` using periodic minimum images.

    ``radius`` and ``d_min`` are in Angstroms; the returned Cartesian position
    is inside the supplied periodic cell or ``None`` after the attempt limit.
    """
    d_min_squared = d_min**2
    for _ in range(max_attempts):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-12
        distance = radius * rng.random() ** (1.0 / 3.0)
        position = centre + direction * distance
        fractional = position @ inv_cell
        fractional -= np.floor(fractional)
        position = fractional @ cell
        distances = _minimum_image_distances(positions, position, cell, inv_cell)
        if len(distances) == 0 or float(np.min(distances)) >= d_min_squared:
            return position
    return None


def _insert_interstitials(
    supercell: Any,
    base: Any,
    n: int,
    settings: PerturbationSettings,
    rng: np.random.RandomState | None,
    elements: Sequence[str],
    d_min: float,
    family: str,
    count_key: str,
    annotate: Annotate,
    *,
    seed: int | None = None,
) -> list[Any]:
    del rng
    base_structure_id = calculate_structure_id(base)
    root_seed = settings.random_seed if seed is None else int(seed)
    cell = np.array(supercell.cell)
    inv_cell = np.linalg.inv(cell)
    output: list[Any] = []
    for index in range(n):
        child_seed = derive_child_seed(base_structure_id, root_seed, family, index)
        slot_rng = np.random.RandomState(child_seed)
        positions = supercell.get_positions().copy()
        new_positions: list[np.ndarray] = []
        new_symbols: list[str] = []
        fraction = slot_rng.uniform(*settings.interstitial_range)
        n_add = max(1, int(fraction * len(supercell)))
        for _ in range(n_add):
            position = find_interstitial_site(
                positions,
                cell,
                inv_cell,
                slot_rng,
                max_attempts=500,
                d_min=d_min,
            )
            if position is not None:
                new_positions.append(position)
                new_symbols.append(elements[slot_rng.randint(len(elements))])
                positions = np.vstack([positions, position])
        result = supercell.copy()
        for position, symbol in zip(new_positions, new_symbols):
            result.append(Atom(symbol=symbol, position=position))
        annotate(
            result,
            base,
            family,
            random_seed=child_seed,
            parameters={
                "requested_n_interstitials": n_add,
                "realised_n_interstitials": len(new_positions),
                count_key: len(new_positions),
            },
            operation_id=f"{family}:{index}",
        )
        output.append(result)
    return output


def _minimum_image_distances(
    positions: np.ndarray,
    position: np.ndarray,
    cell: np.ndarray,
    inv_cell: np.ndarray,
) -> np.ndarray:
    if len(positions) == 0:
        return np.empty(0)
    difference = positions - position
    fractional_difference = difference @ inv_cell
    fractional_difference -= np.rint(fractional_difference)
    cartesian_difference = fractional_difference @ cell
    return np.sum(cartesian_difference * cartesian_difference, axis=1)


__all__ = [
    "find_interstitial_site",
    "find_site_near",
    "gas_in_vacancy",
    "gas_interstitials",
    "interstitials",
    "vacancies",
    "vacancy_interstitial",
]
