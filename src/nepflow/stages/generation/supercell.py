"""Canonical deterministic target-supercell construction."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk

from .generators.composition_primitives import calculate_composition_realization

logger = logging.getLogger(__name__)

PARENT_TOPOLOGY_SCHEMA = "parent-topology-v1"
PARENT_TOPOLOGY_VERSION = 1
DEFAULT_SYMMETRY_TOLERANCE = 1.0e-3
PARENT_TOPOLOGY_ARRAYS = (
    "parent_site_index",
    "parent_orbit_index",
    "parent_cell_translation",
    "parent_unwrapped_fractional",
    "parent_topology_mapped",
)


class ParentTopologyError(ValueError):
    """Raised when trustworthy parent topology cannot be established."""


def ensure_parent_topology(
    atoms: Atoms,
    *,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
) -> Atoms:
    """Return a copy carrying immutable parent-site topology.

    Symmetry is evaluated only for a pristine structure with no existing
    topology.  A structure marked as a derived perturbation, or one carrying
    partial topology metadata, fails explicitly instead of guessing its
    parent magnetic sublattice.
    """

    result = atoms.copy()
    topology_state = _topology_state(result)
    if topology_state == "complete":
        require_parent_topology(result)
        return result
    if topology_state == "partial":
        raise ParentTopologyError("structure carries incomplete parent topology")
    if any(
        key in result.info
        for key in (
            "parent_topology_schema",
            "parent_topology_version",
            "topology_parent_structure_id",
        )
    ):
        raise ParentTopologyError("structure carries incomplete parent topology metadata")
    perturbation_type = result.info.get("perturbation_type")
    if perturbation_type not in (None, "unperturbed"):
        raise ParentTopologyError(
            "cannot infer parent topology from a derived structure without inherited topology: "
            f"perturbation_type={perturbation_type!r}"
        )
    _attach_parent_topology(result, symmetry_tolerance=symmetry_tolerance)
    return result


def require_parent_topology(atoms: Atoms) -> None:
    """Validate that all required parent topology arrays and metadata exist."""

    if _topology_state(atoms) != "complete":
        raise ParentTopologyError("structure does not carry complete parent topology")
    missing_info = [
        key
        for key in (
            "parent_topology_schema",
            "parent_topology_version",
            "topology_parent_structure_id",
            "topology_supercell_repeat",
            "topology_transformation",
            "topology_symmetry_tolerance",
        )
        if key not in atoms.info
    ]
    if missing_info:
        raise ParentTopologyError(
            "structure carries incomplete parent topology metadata: " + ", ".join(missing_info)
        )
    if atoms.info["parent_topology_schema"] != PARENT_TOPOLOGY_SCHEMA:
        raise ParentTopologyError("unsupported parent topology schema")
    try:
        version = int(atoms.info["parent_topology_version"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParentTopologyError("parent topology version is invalid") from exc
    if version != PARENT_TOPOLOGY_VERSION:
        raise ParentTopologyError("unsupported parent topology version")
    repeat = _topology_repeat(atoms)
    if any(value <= 0 for value in repeat):
        raise ParentTopologyError("parent topology repeat must contain positive integers")
    try:
        site_indices = np.asarray(atoms.arrays["parent_site_index"], dtype=int)
        orbit_indices = np.asarray(atoms.arrays["parent_orbit_index"], dtype=int)
        translations = np.asarray(atoms.arrays["parent_cell_translation"], dtype=int)
        mapped = np.asarray(atoms.arrays["parent_topology_mapped"], dtype=bool)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParentTopologyError("parent topology arrays are invalid") from exc
    if (
        site_indices.shape != (len(atoms),)
        or orbit_indices.shape != (len(atoms),)
        or translations.shape != (len(atoms), 3)
        or mapped.shape != (len(atoms),)
    ):
        raise ParentTopologyError("parent topology array lengths or shapes are invalid")
    for name, values in (
        ("parent_site_index", site_indices),
        ("parent_orbit_index", orbit_indices),
    ):
        if np.any(values[mapped] < 0):
            raise ParentTopologyError(f"mapped topology array {name} contains a sentinel")
    coordinates = np.asarray(atoms.arrays["parent_unwrapped_fractional"], dtype=float)
    if coordinates.shape != (len(atoms), 3) or not np.isfinite(coordinates[mapped]).all():
        raise ParentTopologyError("mapped parent fractional coordinates are invalid")
    if not np.isfinite(translations).all():
        raise ParentTopologyError("parent topology translations are invalid")


def mark_added_atoms_unmapped(atoms: Atoms, start_index: int) -> None:
    """Mark atoms appended after ``start_index`` as explicitly unmapped."""

    if _topology_state(atoms) != "complete":
        return
    if not 0 <= start_index <= len(atoms):
        raise ParentTopologyError(f"invalid topology append index: {start_index}")
    _set_topology_rows(
        atoms,
        start_index,
        parent_site_index=np.full(len(atoms) - start_index, -1, dtype=int),
        parent_orbit_index=np.full(len(atoms) - start_index, -1, dtype=int),
        parent_cell_translation=np.full((len(atoms) - start_index, 3), -1, dtype=int),
        parent_unwrapped_fractional=np.full((len(atoms) - start_index, 3), np.nan, dtype=float),
        parent_topology_mapped=np.zeros(len(atoms) - start_index, dtype=bool),
    )


def mark_added_atoms_mapped(
    atoms: Atoms,
    start_index: int,
    rows: Sequence[Mapping[str, Any] | None],
) -> None:
    """Apply supported crystallographic site mappings to appended atoms."""

    if _topology_state(atoms) != "complete":
        return
    if not 0 <= start_index <= len(atoms):
        raise ParentTopologyError(f"invalid topology append index: {start_index}")
    if len(rows) != len(atoms) - start_index:
        raise ParentTopologyError("mapped topology row count does not match appended atoms")
    for offset, row in enumerate(rows):
        index = start_index + offset
        if row is None:
            mark_added_atoms_unmapped(atoms, index)
            continue
        try:
            site_index = int(row["parent_site_index"])
            orbit_index = int(row["parent_orbit_index"])
            translation = tuple(int(value) for value in row["parent_cell_translation"])
            fractional = np.asarray(row["parent_unwrapped_fractional"], dtype=float)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ParentTopologyError(
                "invalid mapped crystallographic interstitial definition"
            ) from exc
        if site_index < 0 or orbit_index < 0 or len(translation) != 3:
            raise ParentTopologyError("mapped crystallographic site contains invalid indices")
        if fractional.shape != (3,) or not np.isfinite(fractional).all():
            raise ParentTopologyError("mapped crystallographic site has invalid coordinates")
        _set_topology_rows(
            atoms,
            index,
            parent_site_index=np.asarray([site_index], dtype=int),
            parent_orbit_index=np.asarray([orbit_index], dtype=int),
            parent_cell_translation=np.asarray([translation], dtype=int),
            parent_unwrapped_fractional=np.asarray([fractional], dtype=float),
            parent_topology_mapped=np.asarray([True], dtype=bool),
        )


def _topology_state(atoms: Atoms) -> str:
    present = [name in atoms.arrays for name in PARENT_TOPOLOGY_ARRAYS]
    if all(present):
        return "complete"
    return "partial" if any(present) else "absent"


def _attach_parent_topology(atoms: Atoms, *, symmetry_tolerance: float) -> None:
    if not np.isfinite(symmetry_tolerance) or symmetry_tolerance <= 0.0:
        raise ValueError("symmetry_tolerance must be finite and positive")
    fully_periodic = all(bool(value) for value in np.asarray(atoms.pbc, dtype=bool).reshape(-1)[:3])
    source_structure_id = _structure_id(atoms)
    scaled_unwrapped = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    n_atoms = len(atoms)
    orbit_assignment_reliable = True
    if n_atoms == 0:
        orbit_indices = np.empty(0, dtype=int)
        space_group = None
        space_group_number = None
    elif (
        not fully_periodic
        or len(np.unique(np.round(atoms.get_scaled_positions(wrap=True), decimals=12), axis=0))
        != n_atoms
    ):
        # Duplicate parent sites are not a valid spglib input.  Retain a
        # conservative, deterministic per-site topology for malformed or
        # non-3D-periodic parents, while recording that no symmetry orbit or
        # space group was inferred.
        orbit_indices = np.arange(n_atoms, dtype=int)
        space_group = None
        space_group_number = None
        orbit_assignment_reliable = False
    else:
        try:
            from pymatgen.core import Lattice, Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            structure = Structure(
                Lattice(np.asarray(atoms.cell.array, dtype=float)),
                atoms.get_chemical_symbols(),
                atoms.get_scaled_positions(wrap=True),
                coords_are_cartesian=False,
                to_unit_cell=True,
                validate_proximity=False,
            )
            analyzer = SpacegroupAnalyzer(structure, symprec=symmetry_tolerance)
            dataset = analyzer.get_symmetry_dataset()
            representatives = np.asarray(dataset.equivalent_atoms, dtype=int)
            unique_representatives = sorted(set(int(value) for value in representatives))
            orbit_lookup = {
                representative: orbit for orbit, representative in enumerate(unique_representatives)
            }
            orbit_indices = np.asarray(
                [orbit_lookup[int(value)] for value in representatives], dtype=int
            )
            space_group = str(analyzer.get_space_group_symbol())
            space_group_number = int(analyzer.get_space_group_number())
        except Exception as exc:
            raise ParentTopologyError(
                f"failed to assign parent crystallographic topology: {type(exc).__name__}: {exc}"
            ) from exc

    _set_topology_arrays(
        atoms,
        parent_site_index=np.arange(n_atoms, dtype=int),
        parent_orbit_index=orbit_indices,
        parent_cell_translation=np.zeros((n_atoms, 3), dtype=int),
        parent_unwrapped_fractional=scaled_unwrapped,
        parent_topology_mapped=np.ones(n_atoms, dtype=bool),
    )
    atoms.info.update(
        {
            "parent_topology_schema": PARENT_TOPOLOGY_SCHEMA,
            "parent_topology_version": PARENT_TOPOLOGY_VERSION,
            "parent_topology_source_structure_id": source_structure_id,
            "topology_parent_structure_id": source_structure_id,
            "parent_structure_id": source_structure_id,
            "topology_supercell_repeat": (1, 1, 1),
            "topology_transformation": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            "topology_symmetry_tolerance": float(symmetry_tolerance),
            "parent_topology_symmetry_tolerance": float(symmetry_tolerance),
            "topology_source_space_group": space_group,
            "parent_topology_source_space_group": space_group,
            "topology_source_space_group_number": space_group_number,
            "topology_orbit_assignment_reliable": orbit_assignment_reliable,
        }
    )


def _set_topology_arrays(atoms: Atoms, **arrays: np.ndarray) -> None:
    for name, values in arrays.items():
        atoms.set_array(name, np.asarray(values))
    if "parent_topology_mapped" in arrays:
        atoms.set_array("topology_mapped", np.asarray(arrays["parent_topology_mapped"], dtype=bool))


def _set_topology_rows(atoms: Atoms, start_index: int, **arrays: np.ndarray) -> None:
    for name, values in arrays.items():
        atoms.arrays[name][start_index:] = values
    if "parent_topology_mapped" in arrays:
        atoms.arrays["topology_mapped"][start_index:] = arrays["parent_topology_mapped"]


def _topology_repeat(atoms: Atoms) -> tuple[int, int, int]:
    value = atoms.info.get("topology_supercell_repeat", (1, 1, 1))
    try:
        repeat = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        raise ParentTopologyError("parent topology repeat is invalid") from None
    if len(repeat) != 3:
        raise ParentTopologyError("parent topology repeat must contain three integers")
    return repeat


def _structure_id(atoms: Atoms) -> str:
    from nepflow.domain.identities import calculate_structure_id

    return calculate_structure_id(atoms)


class CompositionRealizabilityError(ValueError):
    """Raised when no permitted diagonal repeat satisfies composition tolerance."""


def build_target_supercell(
    source: Atoms | str,
    crystal_structure: str | int | None = None,
    target_n_atoms: int | None = None,
    *,
    composition: Mapping[str, float] | None = None,
    composition_tolerance: float | None = 0.05,
    min_n_atoms: int | None = None,
    max_n_atoms: int | None = None,
    raise_on_error: bool = False,
) -> Atoms | None:
    """Build a target supercell using a deterministic diagonal repeat search.

    ``target_n_atoms`` is a preferred size.  When a composition is supplied,
    only candidate sizes whose canonical integer realization is within
    ``composition_tolerance`` are accepted.  The returned structure remains a
    parent lattice for configurational generators; composition assignment is
    deliberately owned by their shared composition primitive.
    """

    target = target_n_atoms
    if isinstance(source, Atoms):
        if target is None and isinstance(crystal_structure, int):
            target = crystal_structure
        if target is None:
            raise TypeError("target_n_atoms is required for an Atoms source")
        return _expand_atoms(
            source,
            target,
            composition=composition,
            composition_tolerance=composition_tolerance,
            min_n_atoms=min_n_atoms,
            max_n_atoms=max_n_atoms,
            raise_on_error=raise_on_error,
        )

    if not isinstance(crystal_structure, str) or target is None:
        raise TypeError("crystal_structure and target_n_atoms are required for an element source")
    return _build_lattice(
        source,
        crystal_structure,
        target,
        composition=composition,
        composition_tolerance=composition_tolerance,
        min_n_atoms=min_n_atoms,
        max_n_atoms=max_n_atoms,
        raise_on_error=raise_on_error,
    )


def _expand_atoms(
    atoms: Atoms,
    target_n_atoms: int,
    *,
    composition: Mapping[str, float] | None,
    composition_tolerance: float | None,
    min_n_atoms: int | None,
    max_n_atoms: int | None,
    raise_on_error: bool,
) -> Atoms:
    """Expand an existing structure through the same repeat planner."""

    atoms = ensure_parent_topology(atoms)
    _validate_search_inputs(
        target_n_atoms,
        composition=composition,
        composition_tolerance=composition_tolerance,
        min_n_atoms=min_n_atoms,
        max_n_atoms=max_n_atoms,
    )
    n_atoms = len(atoms)
    if n_atoms == 0:
        _record_plan_metadata(
            atoms,
            target_n_atoms=target_n_atoms,
            repeat=_topology_repeat(atoms),
            composition=composition,
            composition_tolerance=composition_tolerance,
        )
        return atoms
    if n_atoms >= target_n_atoms and composition is None:
        if max_n_atoms is not None and n_atoms > max_n_atoms:
            raise CompositionRealizabilityError(
                "existing parent exceeds the configured maximum atom count: "
                f"atom_count={n_atoms}, max_n_atoms={max_n_atoms}"
            )
        if min_n_atoms is None or n_atoms >= min_n_atoms:
            _record_plan_metadata(
                atoms,
                target_n_atoms=target_n_atoms,
                repeat=_topology_repeat(atoms),
                composition=composition,
                composition_tolerance=composition_tolerance,
            )
            return atoms
    if n_atoms >= target_n_atoms and composition is not None:
        realization = calculate_composition_realization(composition, n_atoms)
        assert composition_tolerance is not None
        if realization.max_error > composition_tolerance + 1.0e-12:
            repeat = _select_repeat(
                atoms,
                target_n_atoms,
                composition=composition,
                composition_tolerance=composition_tolerance,
                min_n_atoms=min_n_atoms,
                max_n_atoms=max_n_atoms,
            )
            if repeat is None:
                raise CompositionRealizabilityError(
                    "existing parent atom count cannot realize composition within tolerance: "
                    f"atom_count={n_atoms}, error={realization.max_error:.6g}, "
                    f"tolerance={composition_tolerance:.6g}"
                )
            expanded = _repeat(atoms, repeat)
            _record_plan_metadata(
                expanded,
                target_n_atoms=target_n_atoms,
                repeat=_topology_repeat(expanded),
                composition=composition,
                composition_tolerance=composition_tolerance,
            )
            return expanded
        expanded = atoms
        _record_plan_metadata(
            expanded,
            target_n_atoms=target_n_atoms,
            repeat=_topology_repeat(expanded),
            composition=composition,
            composition_tolerance=composition_tolerance,
        )
        return expanded

    repeat = _select_repeat(
        atoms,
        target_n_atoms,
        composition=composition,
        composition_tolerance=composition_tolerance,
        min_n_atoms=min_n_atoms,
        max_n_atoms=max_n_atoms,
    )
    if repeat is None:
        if raise_on_error or min_n_atoms is not None or max_n_atoms is not None:
            raise CompositionRealizabilityError(
                "no allowed diagonal repeat satisfies the requested supercell constraints"
            )
        _record_plan_metadata(
            atoms,
            target_n_atoms=target_n_atoms,
            repeat=_topology_repeat(atoms),
            composition=composition,
            composition_tolerance=composition_tolerance,
        )
        return atoms
    expanded = _repeat(atoms, repeat)
    _record_plan_metadata(
        expanded,
        target_n_atoms=target_n_atoms,
        repeat=_topology_repeat(expanded),
        composition=composition,
        composition_tolerance=composition_tolerance,
    )
    return expanded


def _build_lattice(
    element: str,
    crystal_structure: str,
    target_n_atoms: int,
    *,
    composition: Mapping[str, float] | None,
    composition_tolerance: float | None,
    min_n_atoms: int | None,
    max_n_atoms: int | None,
    raise_on_error: bool,
) -> Atoms | None:
    try:
        if crystal_structure == "hcp":
            base = bulk(element, "hcp", a=3.0, c=3.0 * 1.633)
        else:
            base = bulk(element, crystal_structure, a=3.0)
    except Exception:
        # ASE bulk builders expose several structure-specific exception types.
        # ``None`` remains the optional-generator result for ordinary lattice
        # construction failures when fail-fast behaviour was not requested.
        if raise_on_error:
            raise
        logger.debug("Cannot build %s-%s", element, crystal_structure, exc_info=True)
        return None

    base = ensure_parent_topology(base)
    _validate_search_inputs(
        target_n_atoms,
        composition=composition,
        composition_tolerance=composition_tolerance,
        min_n_atoms=min_n_atoms,
        max_n_atoms=max_n_atoms,
    )
    if len(base) == 0:
        return None
    repeat = _select_repeat(
        base,
        target_n_atoms,
        composition=composition,
        composition_tolerance=composition_tolerance,
        min_n_atoms=min_n_atoms,
        max_n_atoms=max_n_atoms,
    )
    if repeat is None:
        error = CompositionRealizabilityError(
            "no allowed diagonal repeat satisfies the requested supercell constraints: "
            f"target_n_atoms={target_n_atoms}, base_n_atoms={len(base)}, "
            f"composition={dict(composition) if composition is not None else None}, "
            f"tolerance={composition_tolerance}"
        )
        if (
            composition is not None
            or raise_on_error
            or min_n_atoms is not None
            or max_n_atoms is not None
        ):
            raise error
        return None
    expanded = _repeat(base, repeat)
    _record_plan_metadata(
        expanded,
        target_n_atoms=target_n_atoms,
        repeat=_topology_repeat(expanded),
        composition=composition,
        composition_tolerance=composition_tolerance,
    )
    return expanded


def _select_repeat(
    parent: Atoms,
    target_n_atoms: int,
    *,
    composition: Mapping[str, float] | None,
    composition_tolerance: float | None,
    min_n_atoms: int | None,
    max_n_atoms: int | None,
) -> tuple[int, int, int] | None:
    """Select the best repeat with stable scientific tie-breaking."""

    n_base = len(parent)
    preferred_scale = max(target_n_atoms / n_base, 1.0)
    max_allowed = max_n_atoms if max_n_atoms is not None else target_n_atoms * 2 + n_base * 8
    search_scale = max(preferred_scale, max_allowed / n_base)
    max_axis = max(2, int(np.ceil(search_scale ** (1.0 / 3.0))) + 3)
    lengths = np.linalg.norm(np.asarray(parent.cell.array, dtype=float), axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        lengths = np.ones(3, dtype=float)

    repeat_candidates = set(itertools.product(range(1, max_axis + 1), repeat=3))
    if composition is not None:
        # A balanced cube-root neighbourhood does not contain factorizations
        # such as 129 = 3 * 3 * 43.  Add factorizations for a bounded window
        # around the preferred atom count so nearby composition-compatible
        # sizes are considered without scanning an unnecessarily large cube.
        window = max(16, int(np.ceil(target_n_atoms * 0.1)))
        lower = max(1, target_n_atoms - window)
        upper = min(max_allowed, target_n_atoms + window)
        for atom_count in range(lower, upper + 1):
            if atom_count % n_base:
                continue
            repeat_candidates.update(_factor_repeat_triples(atom_count // n_base))

    candidates: list[tuple[tuple[float, ...], tuple[int, int, int]]] = []
    for raw_repeat in sorted(repeat_candidates):
        repeat = (int(raw_repeat[0]), int(raw_repeat[1]), int(raw_repeat[2]))
        atom_count = n_base * int(np.prod(repeat))
        if composition is None and atom_count < target_n_atoms:
            continue
        if min_n_atoms is not None and atom_count < min_n_atoms:
            continue
        if max_n_atoms is not None and atom_count > max_n_atoms:
            continue

        realization = None
        if composition is not None:
            assert composition_tolerance is not None
            realization = calculate_composition_realization(composition, atom_count)
            if realization.max_error > composition_tolerance + 1.0e-12:
                continue

        expanded_lengths = lengths * np.asarray(repeat, dtype=float)
        aspect_penalty = float(np.max(expanded_lengths) / np.min(expanded_lengths) - 1.0)
        if composition is None:
            # For unconstrained parent expansion, preserve a balanced periodic
            # cell before considering atom-count distance.  This retains the
            # historical cubic result while allowing anisotropic parents to
            # select a geometry-aware diagonal repeat.
            score = (
                round(aspect_penalty, 12),
                float(abs(atom_count - target_n_atoms)),
                float(atom_count),
                *(float(value) for value in repeat),
            )
        else:
            assert realization is not None
            score = (
                float(abs(atom_count - target_n_atoms)),
                round(aspect_penalty, 12),
                float(realization.max_error),
                float(atom_count),
                *(float(value) for value in repeat),
            )
        candidates.append((score, repeat))

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate[0])
    return candidates[0][1]


def _factor_repeat_triples(product: int) -> set[tuple[int, int, int]]:
    """Return all ordered positive triples with the requested product."""

    triples: set[tuple[int, int, int]] = set()
    for first in range(1, int(np.sqrt(product)) + 1):
        if product % first:
            continue
        quotient = product // first
        for second in range(1, int(np.sqrt(quotient)) + 1):
            if quotient % second:
                continue
            third = quotient // second
            for permutation in itertools.permutations((first, second, third)):
                triples.add((permutation[0], permutation[1], permutation[2]))
    return triples


def _repeat(atoms: Atoms, repeat: tuple[int, int, int]) -> Atoms:
    require_parent_topology(atoms)
    old_repeat = np.asarray(_topology_repeat(atoms), dtype=int)
    expanded = atoms.repeat(repeat)
    n_source = len(atoms)
    repeat_translations = np.asarray(
        list(itertools.product(*(range(value) for value in repeat))), dtype=int
    )
    source_indices = np.tile(np.arange(n_source, dtype=int), len(repeat_translations))
    translated = np.repeat(repeat_translations, n_source, axis=0)
    translated_in_parent_cells = translated * old_repeat
    _set_topology_arrays(
        expanded,
        parent_site_index=np.asarray(atoms.arrays["parent_site_index"])[source_indices],
        parent_orbit_index=np.asarray(atoms.arrays["parent_orbit_index"])[source_indices],
        parent_cell_translation=(
            np.asarray(atoms.arrays["parent_cell_translation"])[source_indices]
            + translated_in_parent_cells
        ),
        parent_unwrapped_fractional=(
            np.asarray(atoms.arrays["parent_unwrapped_fractional"])[source_indices]
            + translated_in_parent_cells
        ),
        parent_topology_mapped=np.asarray(atoms.arrays["parent_topology_mapped"])[source_indices],
    )
    expanded.pbc = atoms.pbc
    new_repeat = tuple((old_repeat * np.asarray(repeat, dtype=int)).tolist())
    _set_topology_repeat(expanded, new_repeat)
    return expanded


def _record_plan_metadata(
    atoms: Atoms,
    *,
    target_n_atoms: int,
    repeat: tuple[int, int, int],
    composition: Mapping[str, float] | None,
    composition_tolerance: float | None,
) -> None:
    atoms.info["supercell_repeat"] = repeat
    if _topology_state(atoms) == "complete":
        _set_topology_repeat(atoms, repeat)
    atoms.info["supercell_target_n_atoms"] = target_n_atoms
    atoms.info["supercell_realized_n_atoms"] = len(atoms)
    if composition is not None:
        realization = calculate_composition_realization(composition, len(atoms))
        atoms.info["composition_target_counts"] = dict(realization.counts)
        atoms.info["composition_target_fractions"] = dict(realization.realized)
        atoms.info["composition_target_error"] = realization.max_error
        if composition_tolerance is not None:
            atoms.info["composition_tolerance"] = float(composition_tolerance)


def _set_topology_repeat(atoms: Atoms, repeat: tuple[int, int, int]) -> None:
    normalized = tuple(int(value) for value in repeat)
    if len(normalized) != 3 or any(value <= 0 for value in normalized):
        raise ParentTopologyError("parent topology repeat must contain positive integers")
    matrix = tuple(
        tuple(int(row == column) * normalized[row] for column in range(3)) for row in range(3)
    )
    atoms.info["topology_supercell_repeat"] = normalized
    atoms.info["topology_transformation"] = matrix


def _validate_search_inputs(
    target_n_atoms: int,
    *,
    composition: Mapping[str, float] | None,
    composition_tolerance: float | None,
    min_n_atoms: int | None,
    max_n_atoms: int | None,
) -> None:
    if target_n_atoms <= 0:
        raise ValueError("target_n_atoms must be positive")
    if min_n_atoms is not None and min_n_atoms <= 0:
        raise ValueError("min_n_atoms must be positive")
    if max_n_atoms is not None and max_n_atoms <= 0:
        raise ValueError("max_n_atoms must be positive")
    if min_n_atoms is not None and max_n_atoms is not None and min_n_atoms > max_n_atoms:
        raise ValueError("min_n_atoms must not exceed max_n_atoms")
    if composition is not None:
        if composition_tolerance is None:
            raise ValueError("composition_tolerance is required with composition")
        if not np.isfinite(composition_tolerance) or not 0.0 <= composition_tolerance <= 1.0:
            raise ValueError("composition_tolerance must be in [0, 1]")


__all__ = [
    "CompositionRealizabilityError",
    "DEFAULT_SYMMETRY_TOLERANCE",
    "PARENT_TOPOLOGY_ARRAYS",
    "PARENT_TOPOLOGY_SCHEMA",
    "PARENT_TOPOLOGY_VERSION",
    "ParentTopologyError",
    "build_target_supercell",
    "ensure_parent_topology",
    "mark_added_atoms_mapped",
    "mark_added_atoms_unmapped",
    "require_parent_topology",
]
