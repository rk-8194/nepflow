"""Vacancy, interstitial, substitution, and antisite perturbations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from ase import Atom

from nepflow.domain.identities import calculate_structure_id

from ..supercell import mark_added_atoms_mapped, mark_added_atoms_unmapped
from .models import PerturbationSettings, derive_child_seed

Annotate = Callable[..., Any]


def _gas_interstitial_distance(settings: PerturbationSettings) -> float:
    value = settings.gas_interstitial_d_min
    return settings.interstitial_d_min if value is None else value


def _requested_count(
    fraction_range: tuple[float, float],
    n_atoms: int,
    rng: np.random.RandomState,
) -> tuple[int, float]:
    """Return an integer target and the sampled requested concentration."""

    if n_atoms <= 0:
        raise ValueError("cannot realize a defect in an empty structure")
    fraction = rng.uniform(*fraction_range)
    return max(1, int(fraction * n_atoms)), float(fraction)


def _count_parameters(
    family: str,
    requested: int,
    realized: int,
    denominator: int,
    requested_concentration: float | None = None,
) -> dict[str, Any]:
    if requested_concentration is None:
        requested_concentration = requested / denominator if denominator else 0.0
    realized_concentration = realized / denominator if denominator else 0.0
    singular = {
        "vacancies": "vacancy",
        "interstitials": "interstitial",
        "substitutions": "substitution",
        "antisites": "antisite",
    }.get(family, family[:-1] if family.endswith("s") else family)
    return {
        f"requested_n_{family}": requested,
        f"realised_n_{family}": realized,
        f"n_{family}": realized,
        f"requested_{family}_concentration": requested_concentration,
        f"realised_{family}_concentration": realized_concentration,
        f"requested_{singular}_concentration": requested_concentration,
        f"realised_{singular}_concentration": realized_concentration,
        "requested_concentration": requested_concentration,
        "realised_concentration": realized_concentration,
        "requested_defect_count": requested,
        "realised_defect_count": realized,
        "requested_defect_concentration": requested_concentration,
        "realised_defect_concentration": realized_concentration,
    }


def _affected_species(*species: str) -> str:
    return ",".join(dict.fromkeys(str(item) for item in species if item))


def vacancies(
    supercell: Any,
    base: Any,
    n: int,
    settings: PerturbationSettings,
    rng: np.random.RandomState | None,
    annotate: Annotate,
    *,
    seed: int | None = None,
    slot_start: int = 0,
) -> list[Any]:
    """Generate species-restricted vacancy structures with deterministic slots."""

    del rng
    base_structure_id = calculate_structure_id(base)
    root_seed = settings.random_seed if seed is None else int(seed)
    symbols = np.asarray(supercell.get_chemical_symbols())
    allowed = set(settings.vacancy_species)
    eligible = (
        np.flatnonzero(np.isin(symbols, list(allowed))) if allowed else np.arange(len(symbols))
    )
    if allowed and len(eligible) == 0:
        raise ValueError(
            "vacancy_species does not match any atom in the supplied structure: "
            + ", ".join(sorted(allowed))
        )
    if len(symbols) <= 1:
        raise ValueError("cannot realize a vacancy while retaining one atom")

    output: list[Any] = []
    for index in range(slot_start, slot_start + n):
        child_seed = derive_child_seed(base_structure_id, root_seed, "vacancy", index)
        slot_rng = np.random.RandomState(child_seed)
        requested, requested_concentration = _requested_count(
            settings.vacancy_range, len(supercell), slot_rng
        )
        if requested > len(eligible) or requested >= len(supercell):
            raise ValueError(
                "requested vacancy count cannot be realized for the configured species"
            )
        if allowed:
            removed_indices = {
                int(item) for item in slot_rng.choice(eligible, size=requested, replace=False)
            }
            keep = [item for item in range(len(supercell)) if item not in removed_indices]
        else:
            keep = sorted(
                int(item)
                for item in slot_rng.choice(
                    len(supercell), size=len(supercell) - requested, replace=False
                )
            )
            removed_indices = set(range(len(supercell))) - set(keep)
        vacancy = supercell[keep]
        removed_species = sorted({str(symbols[item]) for item in removed_indices})
        parameters = _count_parameters(
            "vacancies",
            requested,
            requested,
            len(supercell),
            requested_concentration,
        )
        parameters.update(
            {
                "vacancy_species": ",".join(sorted(allowed)) if allowed else "all",
                "affected_species": _affected_species(*removed_species),
            }
        )
        annotate(
            vacancy,
            base,
            "vacancy",
            random_seed=child_seed,
            parameters=parameters,
            operation_id=f"vacancy:{index}",
        )
        output.append(vacancy)
    return output


def substitutions(
    supercell: Any,
    base: Any,
    n: int,
    settings: PerturbationSettings,
    rng: np.random.RandomState | None,
    annotate: Annotate,
    *,
    seed: int | None = None,
    slot_start: int = 0,
) -> list[Any]:
    """Replace source species with configured target species."""

    del rng
    if not settings.substitution_pairs:
        raise ValueError("substitution_pairs must be configured for substitution defects")
    base_structure_id = calculate_structure_id(base)
    root_seed = settings.random_seed if seed is None else int(seed)
    symbols = np.asarray(supercell.get_chemical_symbols())
    output: list[Any] = []
    for index in range(slot_start, slot_start + n):
        source, target = settings.substitution_pairs[index % len(settings.substitution_pairs)]
        child_seed = derive_child_seed(base_structure_id, root_seed, "substitution", index)
        slot_rng = np.random.RandomState(child_seed)
        requested, requested_concentration = _requested_count(
            settings.substitution_range, len(supercell), slot_rng
        )
        eligible = np.flatnonzero(symbols == source)
        if len(eligible) < requested:
            raise ValueError(
                f"cannot substitute {requested} {source} atoms; only {len(eligible)} available"
            )
        selected = slot_rng.choice(eligible, size=requested, replace=False)
        result = supercell.copy()
        for atom_index in selected:
            result[int(atom_index)].symbol = target
        parameters = _count_parameters(
            "substitutions",
            requested,
            requested,
            len(supercell),
            requested_concentration,
        )
        parameters.update(
            {
                "source_species": source,
                "target_species": target,
                "affected_species": _affected_species(source, target),
            }
        )
        annotate(
            result,
            base,
            "substitution",
            random_seed=child_seed,
            parameters=parameters,
            operation_id=f"substitution:{index}",
        )
        output.append(result)
    return output


def antisites(
    supercell: Any,
    base: Any,
    n: int,
    settings: PerturbationSettings,
    rng: np.random.RandomState | None,
    annotate: Annotate,
    *,
    seed: int | None = None,
    slot_start: int = 0,
) -> list[Any]:
    """Exchange equal numbers of two configured species."""

    del rng
    if not settings.antisite_pairs:
        raise ValueError("antisite_pairs must be configured for antisite defects")
    base_structure_id = calculate_structure_id(base)
    root_seed = settings.random_seed if seed is None else int(seed)
    symbols = np.asarray(supercell.get_chemical_symbols())
    output: list[Any] = []
    for index in range(slot_start, slot_start + n):
        first, second = settings.antisite_pairs[index % len(settings.antisite_pairs)]
        child_seed = derive_child_seed(base_structure_id, root_seed, "antisite", index)
        slot_rng = np.random.RandomState(child_seed)
        requested, requested_concentration = _requested_count(
            settings.antisite_range, len(supercell), slot_rng
        )
        first_indices = np.flatnonzero(symbols == first)
        second_indices = np.flatnonzero(symbols == second)
        if len(first_indices) < requested or len(second_indices) < requested:
            raise ValueError(
                "cannot realize antisite exchange for "
                f"{first}<->{second} with requested count {requested}"
            )
        selected_first = slot_rng.choice(first_indices, size=requested, replace=False)
        selected_second = slot_rng.choice(second_indices, size=requested, replace=False)
        result = supercell.copy()
        for atom_index in selected_first:
            result[int(atom_index)].symbol = second
        for atom_index in selected_second:
            result[int(atom_index)].symbol = first
        parameters = _count_parameters(
            "antisites",
            requested,
            requested,
            len(supercell),
            requested_concentration,
        )
        parameters.update(
            {
                "source_species": first,
                "target_species": second,
                "affected_species": _affected_species(first, second),
            }
        )
        annotate(
            result,
            base,
            "antisite",
            random_seed=child_seed,
            parameters=parameters,
            operation_id=f"antisite:{index}",
        )
        output.append(result)
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
    slot_start: int = 0,
) -> list[Any]:
    """Generate host interstitials at valid random or configured sites."""

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
        slot_start=slot_start,
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
    slot_start: int = 0,
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
        slot_start=slot_start,
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
    slot_start: int = 0,
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
    cell, inv_cell = _cell_arrays(supercell)
    output: list[Any] = []
    for index in range(slot_start, slot_start + n):
        child_seed = derive_child_seed(base_structure_id, root_seed, "vacancy_interstitial", index)
        slot_rng = np.random.RandomState(child_seed)
        vacancy = supercell.copy()
        requested_vacancies, requested_vacancy_concentration = _requested_count(
            settings.vacancy_range, len(vacancy), slot_rng
        )
        if requested_vacancies >= len(vacancy):
            raise ValueError("cannot realize a vacancy/interstitial state with no host atoms")
        original_symbols = np.asarray(vacancy.get_chemical_symbols())
        allowed = set(settings.vacancy_species)
        eligible = (
            np.flatnonzero(np.isin(original_symbols, list(allowed)))
            if allowed
            else np.arange(len(vacancy))
        )
        if requested_vacancies > len(eligible):
            raise ValueError(
                "requested vacancy count cannot be realized for the configured species"
            )
        if allowed:
            removed_indices = {
                int(item)
                for item in slot_rng.choice(eligible, size=requested_vacancies, replace=False)
            }
            keep = set(range(len(vacancy))) - removed_indices
        else:
            keep = {
                int(item)
                for item in slot_rng.choice(
                    len(vacancy), size=len(vacancy) - requested_vacancies, replace=False
                )
            }
            removed_indices = set(range(len(vacancy))) - keep
        removed_species = [str(original_symbols[item]) for item in sorted(removed_indices)]
        vacancy = vacancy[sorted(keep)]
        positions = vacancy.get_positions().copy()
        defect_positions = np.empty((0, 3), dtype=float)
        new_positions: list[np.ndarray] = []
        new_symbols: list[str] = []
        requested_interstitials, requested_interstitial_concentration = _requested_count(
            settings.interstitial_range, len(supercell), slot_rng
        )
        configured_sites = _configured_interstitial_site_records(settings, cell)
        topology_start = len(vacancy)
        topology_rows: list[Mapping[str, Any] | None] = []
        for insertion_index in range(requested_interstitials):
            element = all_elements[slot_rng.randint(len(all_elements))]
            d_min = (
                _gas_interstitial_distance(settings)
                if element in settings.gas_elements
                else settings.interstitial_d_min
            )
            position = _next_interstitial_position(
                positions,
                defect_positions,
                cell,
                inv_cell,
                slot_rng,
                settings,
                d_min,
                insertion_index,
                configured_sites=configured_sites,
                pbc=supercell.pbc,
            )
            if position is None:
                continue
            new_positions.append(position)
            new_symbols.append(element)
            topology_rows.append(
                configured_sites[insertion_index][1]
                if configured_sites and insertion_index < len(configured_sites)
                else None
            )
            defect_positions = np.vstack([defect_positions, position])
        for position, symbol in zip(new_positions, new_symbols):
            vacancy.append(Atom(symbol=symbol, position=position))
        mark_added_atoms_mapped(vacancy, topology_start, topology_rows)
        parameters: dict[str, Any] = {}
        parameters.update(
            _count_parameters(
                "vacancies",
                requested_vacancies,
                requested_vacancies,
                len(supercell),
                requested_vacancy_concentration,
            )
        )
        parameters.update(
            _count_parameters(
                "interstitials",
                requested_interstitials,
                len(new_positions),
                len(supercell),
                requested_interstitial_concentration,
            )
        )
        parameters["affected_species"] = _affected_species(*removed_species, *new_symbols)
        annotate(
            vacancy,
            base,
            "vacancy_interstitial",
            random_seed=child_seed,
            parameters=parameters,
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
    slot_start: int = 0,
) -> list[Any]:
    """Place configured gas species around a vacancy within the cell."""

    if not settings.gas_elements:
        return []
    del rng
    base_structure_id = calculate_structure_id(base)
    root_seed = settings.random_seed if seed is None else int(seed)
    cell, inv_cell = _cell_arrays(supercell)
    output: list[Any] = []
    for index in range(slot_start, slot_start + n):
        child_seed = derive_child_seed(base_structure_id, root_seed, "gas_in_vacancy", index)
        slot_rng = np.random.RandomState(child_seed)
        vacancy = supercell.copy()
        n_atoms = len(vacancy)
        vacancy_index = slot_rng.randint(n_atoms)
        vacancy_position = vacancy.get_positions()[vacancy_index].copy()
        vacancy_element = vacancy.get_chemical_symbols()[vacancy_index]
        vacancy = vacancy[[item for item in range(n_atoms) if item != vacancy_index]]
        topology_start = len(vacancy)
        requested_gas = slot_rng.randint(1, settings.max_gas_occupancy + 1)
        remaining_positions = vacancy.get_positions().copy()
        defect_positions = np.empty((0, 3), dtype=float)
        placed = 0
        gas_species: list[str] = []
        gas_d_min = _gas_interstitial_distance(settings)
        for gas_index in range(requested_gas):
            gas_element = settings.gas_elements[slot_rng.randint(len(settings.gas_elements))]
            if gas_index == 0:
                offset = slot_rng.normal(scale=0.1, size=3)
                candidate: np.ndarray | None = _wrap_position(
                    vacancy_position + offset, cell, inv_cell
                )
                if not _position_is_valid(
                    candidate,
                    remaining_positions,
                    defect_positions,
                    cell,
                    inv_cell,
                    gas_d_min,
                    settings.defect_defect_d_min,
                    settings.periodic_image_d_min,
                    vacancy.pbc,
                ):
                    candidate = None
                position = candidate
            else:
                position = find_site_near(
                    vacancy_position,
                    remaining_positions,
                    cell,
                    inv_cell,
                    slot_rng,
                    radius=2.5,
                    d_min=gas_d_min,
                    max_attempts=settings.interstitial_max_attempts,
                    defect_positions=defect_positions,
                    defect_defect_d_min=settings.defect_defect_d_min,
                    periodic_image_d_min=settings.periodic_image_d_min,
                    pbc=vacancy.pbc,
                )
            if position is not None:
                vacancy.append(Atom(symbol=gas_element, position=position))
                remaining_positions = np.vstack([remaining_positions, position])
                defect_positions = np.vstack([defect_positions, position])
                placed += 1
                gas_species.append(gas_element)
        mark_added_atoms_unmapped(vacancy, topology_start)
        parameters = {
            "requested_n_vacancies": 1,
            "realised_n_vacancies": 1,
            "n_vacancies": 1,
            "requested_vacancy_concentration": 1 / n_atoms if n_atoms else 0.0,
            "realised_vacancy_concentration": 1 / n_atoms if n_atoms else 0.0,
            "requested_n_gas_atoms": requested_gas,
            "realised_n_gas_atoms": placed,
            "n_gas_atoms": placed,
            "requested_defect_count": requested_gas + 1,
            "realised_defect_count": placed + 1,
            "requested_defect_concentration": (requested_gas + 1) / n_atoms if n_atoms else 0.0,
            "realised_defect_concentration": (placed + 1) / n_atoms if n_atoms else 0.0,
            "requested_gas_concentration": requested_gas / n_atoms if n_atoms else 0.0,
            "realised_gas_concentration": placed / n_atoms if n_atoms else 0.0,
            "vacancy_element": vacancy_element,
            "gas_species": ",".join(gas_species),
            "affected_species": _affected_species(vacancy_element, *gas_species),
        }
        annotate(
            vacancy,
            base,
            "gas_in_vacancy",
            random_seed=child_seed,
            parameters=parameters,
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
    defect_positions: np.ndarray | None = None,
    defect_defect_d_min: float = 0.0,
    periodic_image_d_min: float = 0.0,
    pbc: Any = True,
) -> np.ndarray | None:
    """Find a random fractional-cell site satisfying all separation rules."""

    for _ in range(max(0, max_attempts)):
        fractional = rng.random(3)
        position = fractional @ cell
        if _position_is_valid(
            position,
            positions,
            np.empty((0, 3)) if defect_positions is None else defect_positions,
            cell,
            inv_cell,
            d_min,
            defect_defect_d_min,
            periodic_image_d_min,
            pbc,
        ):
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
    defect_positions: np.ndarray | None = None,
    defect_defect_d_min: float = 0.0,
    periodic_image_d_min: float = 0.0,
    pbc: Any = True,
) -> np.ndarray | None:
    """Find a wrapped site near ``centre`` satisfying periodic separation rules."""

    for _ in range(max(0, max_attempts)):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-12
        distance = radius * rng.random() ** (1.0 / 3.0)
        position = _wrap_position(centre + direction * distance, cell, inv_cell)
        if _position_is_valid(
            position,
            positions,
            np.empty((0, 3)) if defect_positions is None else defect_positions,
            cell,
            inv_cell,
            d_min,
            defect_defect_d_min,
            periodic_image_d_min,
            pbc,
        ):
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
    slot_start: int = 0,
) -> list[Any]:
    del rng
    if not elements:
        raise ValueError(f"{family} requires at least one interstitial species")
    base_structure_id = calculate_structure_id(base)
    root_seed = settings.random_seed if seed is None else int(seed)
    cell, inv_cell = _cell_arrays(supercell)
    configured_sites = _configured_interstitial_site_records(settings, cell)
    output: list[Any] = []
    for index in range(slot_start, slot_start + n):
        child_seed = derive_child_seed(base_structure_id, root_seed, family, index)
        slot_rng = np.random.RandomState(child_seed)
        positions = supercell.get_positions().copy()
        defect_positions = np.empty((0, 3), dtype=float)
        new_positions: list[np.ndarray] = []
        new_symbols: list[str] = []
        topology_rows: list[Mapping[str, Any] | None] = []
        requested, requested_concentration = _requested_count(
            settings.interstitial_range, len(supercell), slot_rng
        )
        for insertion_index in range(requested):
            position = _next_interstitial_position(
                positions,
                defect_positions,
                cell,
                inv_cell,
                slot_rng,
                settings,
                d_min,
                insertion_index,
                configured_sites=configured_sites,
                pbc=supercell.pbc,
            )
            if position is None:
                continue
            new_positions.append(position)
            new_symbols.append(str(elements[int(slot_rng.randint(len(elements)))]))
            topology_rows.append(
                configured_sites[insertion_index][1]
                if configured_sites and insertion_index < len(configured_sites)
                else None
            )
            defect_positions = np.vstack([defect_positions, position])
        result = supercell.copy()
        topology_start = len(result)
        for position, symbol in zip(new_positions, new_symbols):
            result.append(Atom(symbol=symbol, position=position))
        mark_added_atoms_mapped(result, topology_start, topology_rows)
        parameters = _count_parameters(
            "interstitials",
            requested,
            len(new_positions),
            len(supercell),
            requested_concentration,
        )
        parameters.update(
            {
                count_key: len(new_positions),
                "affected_species": _affected_species(*new_symbols),
                "interstitial_site_mode": "crystallographic" if configured_sites else "stochastic",
            }
        )
        annotate(
            result,
            base,
            family,
            random_seed=child_seed,
            parameters=parameters,
            operation_id=f"{family}:{index}",
        )
        output.append(result)
    return output


def _next_interstitial_position(
    positions: np.ndarray,
    defect_positions: np.ndarray,
    cell: np.ndarray,
    inv_cell: np.ndarray,
    rng: np.random.RandomState,
    settings: PerturbationSettings,
    d_min: float,
    insertion_index: int,
    configured_sites: Sequence[tuple[np.ndarray, Mapping[str, Any] | None]] | None = None,
    pbc: Any = True,
) -> np.ndarray | None:
    if configured_sites:
        if insertion_index >= len(configured_sites):
            return None
        candidate = configured_sites[insertion_index][0]
        return (
            candidate
            if _position_is_valid(
                candidate,
                positions,
                defect_positions,
                cell,
                inv_cell,
                d_min,
                settings.defect_defect_d_min,
                settings.periodic_image_d_min,
                pbc,
            )
            else None
        )
    return find_interstitial_site(
        positions,
        cell,
        inv_cell,
        rng,
        max_attempts=settings.interstitial_max_attempts,
        d_min=d_min,
        defect_positions=defect_positions,
        defect_defect_d_min=settings.defect_defect_d_min,
        periodic_image_d_min=settings.periodic_image_d_min,
    )


def _configured_interstitial_positions(
    settings: PerturbationSettings,
    cell: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return tuple(position for position, _ in _configured_interstitial_site_records(settings, cell))


def _configured_interstitial_site_records(
    settings: PerturbationSettings,
    cell: np.ndarray,
) -> tuple[tuple[np.ndarray, Mapping[str, Any] | None], ...]:
    result: list[tuple[np.ndarray, Mapping[str, Any] | None]] = []
    for site in settings.crystallographic_interstitial_sites or settings.interstitial_sites:
        if isinstance(site, Mapping):
            coordinates = site.get("fractional", site.get("position"))
        else:
            coordinates = site
        if (
            not isinstance(coordinates, Sequence)
            or isinstance(coordinates, str)
            or len(coordinates) != 3
        ):
            raise ValueError("interstitial sites must contain three fractional coordinates")
        fractional = np.asarray(coordinates, dtype=float)
        if (
            not np.isfinite(fractional).all()
            or np.any(fractional < 0.0)
            or np.any(fractional >= 1.0)
        ):
            raise ValueError("interstitial sites must use fractional coordinates in [0, 1)")
        result.append((fractional @ cell, _interstitial_topology_metadata(site, fractional)))
    return tuple(result)


def _interstitial_topology_metadata(
    site: Mapping[str, Any] | Any,
    fractional: np.ndarray,
) -> Mapping[str, Any] | None:
    if not isinstance(site, Mapping):
        return None
    if "parent_site_index" not in site or "parent_orbit_index" not in site:
        return None
    translation_value = site.get("parent_cell_translation", site.get("translation", (0, 0, 0)))
    try:
        translation = tuple(int(value) for value in translation_value)
        parent_site_index = int(site["parent_site_index"])
        parent_orbit_index = int(site["parent_orbit_index"])
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise ValueError("mapped interstitial topology metadata is invalid") from exc
    if len(translation) != 3 or parent_site_index < 0 or parent_orbit_index < 0:
        raise ValueError("mapped interstitial topology metadata is invalid")
    unwrapped_value = site.get(
        "parent_unwrapped_fractional",
        site.get("unwrapped_parent_fractional", site.get("unwrapped_fractional")),
    )
    unwrapped = (
        np.asarray(unwrapped_value, dtype=float)
        if unwrapped_value is not None
        else fractional + np.asarray(translation, dtype=float)
    )
    if unwrapped.shape != (3,) or not np.isfinite(unwrapped).all():
        raise ValueError("mapped interstitial topology coordinates are invalid")
    return {
        "parent_site_index": parent_site_index,
        "parent_orbit_index": parent_orbit_index,
        "parent_cell_translation": translation,
        "parent_unwrapped_fractional": unwrapped,
    }


def _cell_arrays(atoms: Any) -> tuple[np.ndarray, np.ndarray]:
    cell = np.asarray(atoms.cell, dtype=float)
    return cell, np.linalg.inv(cell)


def _wrap_position(position: np.ndarray, cell: np.ndarray, inv_cell: np.ndarray) -> np.ndarray:
    fractional = np.asarray(position, dtype=float) @ inv_cell
    fractional -= np.floor(fractional)
    return fractional @ cell


def _position_is_valid(
    position: np.ndarray,
    positions: np.ndarray,
    defect_positions: np.ndarray,
    cell: np.ndarray,
    inv_cell: np.ndarray,
    d_min: float,
    defect_defect_d_min: float,
    periodic_image_d_min: float,
    pbc: Any,
) -> bool:
    distances = _minimum_image_distances(positions, position, cell, inv_cell, pbc=pbc)
    if len(distances) and float(np.min(distances)) < d_min**2:
        return False
    defect_distances = _minimum_image_distances(defect_positions, position, cell, inv_cell, pbc=pbc)
    if len(defect_distances) and float(np.min(defect_distances)) < defect_defect_d_min**2:
        return False
    return _periodic_image_distance(cell, pbc) >= periodic_image_d_min


def _periodic_image_distance(cell: np.ndarray, pbc: Any) -> float:
    flags = np.asarray(pbc, dtype=bool)
    if flags.ndim == 0:
        flags = np.repeat(flags, 3)
    vectors = cell[flags.reshape(-1)[:3]]
    if len(vectors) == 0:
        return float("inf")
    shortest = float("inf")
    for coefficients in np.ndindex(*(3,) * len(vectors)):
        offset = np.asarray(coefficients, dtype=int) - 1
        if not np.any(offset):
            continue
        shortest = min(shortest, float(np.linalg.norm(offset @ vectors)))
    return shortest


def _minimum_image_distances(
    positions: np.ndarray,
    position: np.ndarray,
    cell: np.ndarray,
    inv_cell: np.ndarray,
    pbc: Any = True,
) -> np.ndarray:
    if len(positions) == 0:
        return np.empty(0)
    difference = positions - position
    fractional_difference = difference @ inv_cell
    flags = np.asarray(pbc, dtype=bool)
    if flags.ndim == 0:
        flags = np.repeat(flags, 3)
    flags = flags.reshape(-1)[:3]
    fractional_difference[:, flags] -= np.rint(fractional_difference[:, flags])
    cartesian_difference = fractional_difference @ cell
    return np.sum(cartesian_difference * cartesian_difference, axis=1)


__all__ = [
    "antisites",
    "find_interstitial_site",
    "find_site_near",
    "gas_in_vacancy",
    "gas_interstitials",
    "interstitials",
    "substitutions",
    "vacancies",
    "vacancy_interstitial",
]
