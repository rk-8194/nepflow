"""Deterministic crystalline surface and slab construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from ase import Atoms

from nepflow.domain.identities import calculate_structure_id

from .models import PerturbationSettings, derive_child_seed


class SurfaceConstructionError(ValueError):
    """Raised when a requested surface cannot be constructed exactly."""


def surfaces(
    parent: Any,
    base: Any,
    count: int,
    settings: PerturbationSettings,
    rng: Any,
    annotate: Callable[..., Any],
    *,
    seed: int | None = None,
) -> list[Any]:
    """Build deterministic, validated slabs for the configured Miller indices.

    Pymatgen supplies the crystallographic termination enumeration while ASE
    remains the canonical output representation.  ``count`` is a global upper
    bound over the ordered ``(Miller index, termination)`` stream; asking for
    more candidates than the configured construction can provide is an
    explicit error rather than a duplicated or substituted surface.
    """

    del rng  # Surface construction is deterministic; the seed is provenance only.
    if count <= 0:
        return []
    miller_indices = tuple(_normalise_miller(index) for index in settings.surface_miller_indices)
    if not miller_indices:
        raise SurfaceConstructionError("surface_miller_indices must contain at least one index")

    parent_structure_id = calculate_structure_id(base)
    generated: list[Any] = []
    for miller_index in miller_indices:
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
            if len(generated) == count:
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
        minimum_slab_size = (
            float(settings.surface_thickness)
            if settings.surface_thickness is not None
            else float(settings.surface_layers)
        )
        generator = SlabGenerator(
            structure,
            miller_index,
            min_slab_size=minimum_slab_size,
            min_vacuum_size=float(settings.surface_vacuum),
            center_slab=True,
            primitive=False,
            in_unit_planes=settings.surface_thickness is None,
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
    species = tuple(str(site.specie) for site in slab)
    coordinates = tuple(
        float(value)
        for row in np.round(np.asarray(slab.frac_coords, dtype=float), decimals=10)
        for value in row
    )
    return species, coordinates


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
    layer_count = _layer_count(candidate)
    return {
        "parent_structure_id": calculate_structure_id(base),
        "surface_miller_index": miller_index,
        "miller_index": miller_index,
        "surface_termination": f"termination_{termination_index}",
        "termination": f"termination_{termination_index}",
        "surface_layers": layer_count,
        "surface_requested_layers": int(settings.surface_layers),
        "surface_thickness": settings.surface_thickness,
        "surface_slab_thickness": _slab_thickness(candidate),
        "surface_dimensions": tuple(
            float(np.linalg.norm(np.asarray(candidate.cell, dtype=float)[index]))
            for index in range(3)
        ),
        "surface_vacuum": float(settings.surface_vacuum),
        "surface_pbc": (True, True, False),
        "surface_in_plane_repeat": realized_repeat,
        "surface_termination_policy": settings.surface_termination_policy,
        "surface_symmetric": bool(settings.surface_symmetric),
        "surface_stoichiometry_change": composition_delta,
        "surface_composition_change": composition_delta,
        "surface_atom_count_change": count_delta,
        "surface_stoichiometry_changed": bool(composition_delta),
    }


def _layer_count(candidate: Atoms) -> int:
    scaled = np.asarray(candidate.get_scaled_positions(wrap=False), dtype=float)
    values = sorted(float(value) for value in scaled[:, 2])
    layers: list[float] = []
    for value in values:
        if not layers or abs(value - layers[-1]) > 1.0e-6:
            layers.append(value)
    return len(layers)


def _slab_thickness(candidate: Atoms) -> float:
    positions = np.asarray(candidate.get_positions(), dtype=float)
    return float(np.ptp(positions[:, 2])) if len(positions) else 0.0


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
