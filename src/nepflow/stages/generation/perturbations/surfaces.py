"""Deterministic crystalline surface and slab construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from ase import Atoms

from nepflow.config.models import SUPPORTED_SURFACE_MILLER_INDICES
from nepflow.domain.identities import calculate_structure_id

from .models import PerturbationSettings, derive_child_seed


class SurfaceConstructionError(ValueError):
    """Raised when a requested surface cannot be constructed exactly."""


_SURFACE_PLANNER_VERSION = "phase6-surface-reference-vacuum-v1"
_SURFACE_BACKEND = "pymatgen.SlabGenerator"
_IDENTITY_TRANSFORM = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def surfaces(
    parent: Any,
    base: Any,
    count: int | None,
    settings: PerturbationSettings,
    rng: Any,
    annotate: Callable[..., Any],
    *,
    seed: int | None = None,
) -> list[Any]:
    """Build deterministic, validated slabs for the configured Miller indices.

    Pymatgen supplies the crystallographic termination enumeration while ASE
    remains the canonical output representation. ``count`` is retained as a
    compatibility limit for direct legacy callers; the configured coordinator
    passes ``None`` so every requested orientation and termination is emitted.
    """

    del rng  # Surface construction is deterministic; the seed is provenance only.
    if count is not None and count <= 0:
        return []
    miller_indices = tuple(_normalise_miller(index) for index in settings.surface_miller_indices)
    if not miller_indices:
        raise SurfaceConstructionError("surface_miller_indices must contain at least one index")

    reference_cell = _establish_reference_basis(parent)
    parent_structure_id = calculate_structure_id(base)
    generated: list[Any] = []
    for miller_index in miller_indices:
        if miller_index not in SUPPORTED_SURFACE_MILLER_INDICES:
            raise SurfaceConstructionError(
                f"unsupported Miller index {miller_index}; automatic surfaces support "
                "only (1, 0, 0), (1, 1, 0), and (1, 1, 1)"
            )
        slabs = _enumerate_slabs(parent, miller_index, settings)
        if settings.surface_termination_policy == "first":
            slabs = slabs[:1]
        if settings.surface_max_terminations > 0:
            slabs = slabs[: settings.surface_max_terminations]
        if not slabs:
            raise SurfaceConstructionError(
                f"no supported termination was constructed for Miller index {miller_index}"
            )
        for termination_index, slab in enumerate(slabs):
            candidate, realized_repeat = _canonicalize_slab(slab, settings)
            parameters = _surface_parameters(
                candidate,
                base,
                reference_cell=reference_cell,
                miller_index=miller_index,
                termination_index=termination_index,
                settings=settings,
                realized_repeat=realized_repeat,
            )
            child_seed = derive_child_seed(
                parent_structure_id,
                settings.random_seed if seed is None else seed,
                "surface",
                f"{miller_index}:{termination_index}",
            )
            annotate(
                candidate,
                base,
                "surface",
                random_seed=child_seed,
                parameters=parameters,
                operation_id=(
                    f"surface:{_format_miller(miller_index)}:termination_{termination_index}"
                ),
            )
            generated.append(candidate)
            if count is not None and len(generated) == count:
                return generated

    if count is None:
        return generated

    raise SurfaceConstructionError(
        "requested "
        f"{count} surfaces, but only {len(generated)} configured Miller/termination "
        "candidates were constructible"
    )


def _enumerate_slabs(
    parent: Any,
    miller_index: tuple[int, int, int],
    settings: PerturbationSettings,
) -> list[Any]:
    """Construct and deterministically order all supported terminations."""

    try:
        from pymatgen.core.surface import SlabGenerator
        from pymatgen.io.ase import AseAtomsAdaptor

        if settings.surface_thickness is None and settings.surface_layers <= 0:
            raise SurfaceConstructionError(
                "surface_layers must be positive when surface_thickness is not set"
            )
        structure = AseAtomsAdaptor.get_structure(parent)
        minimum_slab_size = _minimum_slab_size_angstrom(structure, miller_index, settings)
        generator = SlabGenerator(
            structure,
            miller_index,
            min_slab_size=minimum_slab_size,
            min_vacuum_size=float(settings.surface_vacuum),
            center_slab=True,
            primitive=False,
            # Slab depth is converted to Angstroms above.  Vacuum is always
            # passed independently as a physical Angstrom quantity.
            in_unit_planes=False,
            reorient_lattice=True,
        )
        slabs = generator.get_slabs(
            symmetrize=bool(settings.surface_symmetric),
            filter_out_sym_slabs=False,
        )
    except Exception as exc:
        raise SurfaceConstructionError(
            f"failed to construct Miller index {miller_index}: {type(exc).__name__}: {exc}"
        ) from exc
    if not slabs:
        return []
    # Pymatgen's ordering is an implementation detail.  Sort on species and
    # rounded fractional coordinates so serial and parallel runs agree.
    return sorted(slabs, key=_slab_sort_key)


def _slab_sort_key(slab: Any) -> tuple[Any, ...]:
    lattice = tuple(
        float(value)
        for row in np.round(np.asarray(slab.lattice.matrix, dtype=float), decimals=10)
        for value in row
    )
    species = tuple(str(site.specie) for site in slab)
    coordinates = tuple(
        float(value)
        for row in np.round(np.asarray(slab.frac_coords, dtype=float), decimals=10)
        for value in row
    )
    return lattice, species, coordinates


def _establish_reference_basis(parent: Any) -> np.ndarray:
    """Return the explicit parent-cell basis used for Miller interpretation."""

    try:
        cell = np.asarray(parent.cell, dtype=float)
        pbc = np.asarray(parent.pbc, dtype=bool)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SurfaceConstructionError(
            "failed to construct surface: cannot establish an unambiguous "
            "parent crystallographic reference basis"
        ) from exc
    if cell.shape != (3, 3) or pbc.shape != (3,) or not bool(np.all(pbc)):
        raise SurfaceConstructionError(
            "failed to construct surface: parent crystallographic reference basis "
            "requires a fully periodic 3x3 cell"
        )
    if not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) <= 1.0e-10:
        raise SurfaceConstructionError(
            "failed to construct surface: parent crystallographic reference basis "
            "is singular or non-finite"
        )
    return cell


def _minimum_slab_size_angstrom(
    structure: Any,
    miller_index: tuple[int, int, int],
    settings: PerturbationSettings,
) -> float:
    """Convert the legacy layer control into a physical slab-depth target."""

    if settings.surface_thickness is not None:
        return float(settings.surface_thickness)
    try:
        spacing = float(structure.lattice.d_hkl(miller_index))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise SurfaceConstructionError(
            f"failed to determine physical spacing for Miller index {miller_index}"
        ) from exc
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise SurfaceConstructionError(
            f"failed to determine physical spacing for Miller index {miller_index}"
        )
    return float(settings.surface_layers) * spacing


def _canonicalize_slab(
    slab: Any,
    settings: PerturbationSettings,
) -> tuple[Atoms, tuple[int, int]]:
    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        candidate: Any = AseAtomsAdaptor.get_atoms(slab)
    except Exception as exc:
        raise SurfaceConstructionError(
            f"failed to convert constructed slab to ASE: {type(exc).__name__}: {exc}"
        ) from exc
    candidate.set_pbc((True, True, False))
    repeat = _required_repeat(candidate, settings)
    if repeat != (1, 1):
        candidate = candidate.repeat((*repeat, 1))
    candidate.set_pbc((True, True, False))
    return candidate, repeat


def _required_repeat(candidate: Any, settings: PerturbationSettings) -> tuple[int, int]:
    repeat = tuple(int(value) for value in settings.surface_in_plane_repeat)
    minimums = tuple(float(value) for value in settings.surface_min_in_plane_dimensions)
    if len(repeat) != 2 or len(minimums) != 2:
        raise SurfaceConstructionError("surface in-plane controls must contain two values")
    realized = [repeat[0], repeat[1]]
    cell = np.asarray(candidate.cell, dtype=float)
    lengths = [float(np.linalg.norm(cell[index])) for index in (0, 1)]
    for index, minimum in enumerate(minimums):
        if minimum <= 0.0:
            continue
        if lengths[index] <= 1.0e-12:
            raise SurfaceConstructionError("constructed slab has a degenerate in-plane cell")
        realized[index] = max(realized[index], int(np.ceil(minimum / lengths[index])))
    return realized[0], realized[1]


def _surface_parameters(
    candidate: Atoms,
    base: Any,
    *,
    reference_cell: np.ndarray,
    miller_index: tuple[int, int, int],
    termination_index: int,
    settings: PerturbationSettings,
    realized_repeat: tuple[int, int],
) -> dict[str, Any]:
    parent_counts = Counter(base.get_chemical_symbols())
    candidate_counts = Counter(candidate.get_chemical_symbols())
    elements = sorted(set(parent_counts) | set(candidate_counts))
    parent_total = max(1, len(base))
    candidate_total = max(1, len(candidate))
    composition_delta = {
        element: candidate_counts[element] / candidate_total - parent_counts[element] / parent_total
        for element in elements
        if candidate_counts[element] / candidate_total - parent_counts[element] / parent_total
        != 0.0
    }
    count_delta = {
        element: candidate_counts[element] - parent_counts[element]
        for element in elements
        if candidate_counts[element] != parent_counts[element]
    }
    normal, thickness, normal_period, vacuum, in_plane_area = _surface_geometry(candidate)
    requested_vacuum = float(settings.surface_vacuum)
    if not np.isfinite(vacuum) or vacuum + 1.0e-6 < requested_vacuum:
        raise SurfaceConstructionError(
            f"realized vacuum {vacuum:.8f} Angstrom is below requested "
            f"{requested_vacuum:.8f} Angstrom for Miller index {miller_index}"
        )
    reference_normal = _surface_normal(reference_cell, miller_index)
    layer_count = _layer_count(candidate, normal)
    reference_cell_tuple = tuple(tuple(float(value) for value in row) for row in reference_cell)
    backend_version = _backend_version()
    return {
        "parent_structure_id": calculate_structure_id(base),
        "surface_state": "pristine",
        "surface_miller_index": miller_index,
        "miller_index": miller_index,
        "surface_termination": f"termination_{termination_index}",
        "termination": f"termination_{termination_index}",
        "surface_termination_identity": (
            f"{_format_miller(miller_index)}:termination_{termination_index}"
        ),
        "surface_layers": layer_count,
        "surface_requested_layers": int(settings.surface_layers),
        "surface_thickness": settings.surface_thickness,
        "surface_slab_thickness": thickness,
        "surface_material_thickness": thickness,
        "surface_half_depth": 0.5 * thickness,
        "surface_dimensions": tuple(
            float(np.linalg.norm(np.asarray(candidate.cell, dtype=float)[index]))
            for index in range(3)
        ),
        "surface_in_plane_area": in_plane_area,
        "surface_normal_period": normal_period,
        "surface_normal": tuple(float(value) for value in normal),
        "surface_reference_normal": tuple(float(value) for value in reference_normal),
        "surface_reference_basis": "parent_stored_cell",
        "surface_reference_basis_cell": reference_cell_tuple,
        "surface_reference_cell": reference_cell_tuple,
        "surface_parent_to_reference_transformation": _IDENTITY_TRANSFORM,
        "surface_requested_vacuum": requested_vacuum,
        "surface_requested_vacuum_angstrom": requested_vacuum,
        # Keep the historical key as the requested value; the realised value
        # is explicit and independently measured below.
        "surface_vacuum": requested_vacuum,
        "surface_realized_vacuum": vacuum,
        "surface_realized_vacuum_angstrom": vacuum,
        "surface_pbc": (True, True, False),
        "surface_in_plane_repeat": realized_repeat,
        "surface_termination_policy": settings.surface_termination_policy,
        "surface_symmetric": bool(settings.surface_symmetric),
        "surface_stoichiometry_change": composition_delta,
        "surface_composition_change": composition_delta,
        "surface_atom_count_change": count_delta,
        "surface_stoichiometry_changed": bool(composition_delta),
        "surface_backend": _SURFACE_BACKEND,
        "surface_backend_version": backend_version,
        "surface_planner_version": _SURFACE_PLANNER_VERSION,
    }


def _surface_geometry(candidate: Atoms) -> tuple[np.ndarray, float, float, float, float]:
    """Measure normal, material thickness, repeat, vacuum, and area."""

    cell = np.asarray(candidate.cell, dtype=float)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise SurfaceConstructionError("constructed slab has an invalid cell")
    in_plane_cross = np.cross(cell[0], cell[1])
    in_plane_area = float(np.linalg.norm(in_plane_cross))
    if in_plane_area <= 1.0e-12:
        raise SurfaceConstructionError("constructed slab has a degenerate in-plane cell")
    normal = in_plane_cross / in_plane_area
    positions = np.asarray(candidate.get_positions(), dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise SurfaceConstructionError("constructed slab has no measurable material extent")
    projected = positions @ normal
    thickness = float(np.max(projected) - np.min(projected))
    normal_period = abs(float(np.linalg.det(cell))) / in_plane_area
    vacuum = normal_period - thickness
    return normal, thickness, normal_period, vacuum, in_plane_area


def _surface_normal(cell: np.ndarray, miller_index: tuple[int, int, int]) -> np.ndarray:
    """Return the Cartesian normal for an hkl in a row-vector cell basis."""

    reciprocal = np.linalg.solve(cell, np.asarray(miller_index, dtype=float))
    norm = float(np.linalg.norm(reciprocal))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise SurfaceConstructionError("cannot establish a finite surface normal")
    return reciprocal / norm


def _layer_count(candidate: Atoms, normal: np.ndarray) -> int:
    positions = np.asarray(candidate.get_positions(), dtype=float)
    values = sorted(float(value) for value in positions @ normal)
    layers: list[float] = []
    for value in values:
        if not layers or abs(value - layers[-1]) > 1.0e-6:
            layers.append(value)
    return len(layers)


def _backend_version() -> str:
    try:
        return version("pymatgen")
    except PackageNotFoundError:
        return "unknown"


def _normalise_miller(index: Iterable[Any]) -> tuple[int, int, int]:
    values = tuple(index)
    if len(values) != 3:
        raise SurfaceConstructionError("Miller indices must contain exactly three integers")
    try:
        normalized = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise SurfaceConstructionError("Miller indices must contain integers") from exc
    if any(str(value) != str(raw).strip() for value, raw in zip(normalized, values)):
        raise SurfaceConstructionError("Miller indices must contain integers")
    if not any(normalized):
        raise SurfaceConstructionError("Miller index (0, 0, 0) is invalid")
    return normalized  # type: ignore[return-value]


def _format_miller(index: tuple[int, int, int]) -> str:
    return ",".join(str(value) for value in index)


__all__ = ["SurfaceConstructionError", "surfaces"]
