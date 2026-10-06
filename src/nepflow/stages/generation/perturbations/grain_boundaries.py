"""Deterministic construction of the supported crystalline grain boundary."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
from ase import Atoms

from nepflow.domain.identities import calculate_structure_id

from .models import PerturbationSettings, derive_child_seed

SUPPORTED_ROTATION_AXIS = (0, 0, 1)
SUPPORTED_MISORIENTATION_ANGLE = 36.86989764584402
SUPPORTED_SIGMA = 5
SUPPORTED_BOUNDARY_PLANE = (2, 1, 0)


class GrainBoundaryConstructionError(ValueError):
    """Raised when the explicit supported relationship cannot be constructed."""


def grain_boundaries(
    parent: Any,
    base: Any,
    count: int,
    settings: PerturbationSettings,
    rng: Any,
    annotate: Callable[..., Any],
    *,
    seed: int | None = None,
) -> list[Any]:
    """Construct one explicit Sigma-5 [001] symmetric tilt grain boundary.

    The Pymatgen ``GrainBoundaryGenerator`` path is intentionally limited to
    one reproducible relationship.  Pymatgen's ``rm_ratio`` is used as the
    configured overlap tolerance; a second no-removal construction provides
    exact accounting for the atoms removed by that operation.
    """

    del rng
    if count <= 0:
        return []
    if count != 1:
        raise GrainBoundaryConstructionError(
            "the supported grain-boundary parameterisation produces exactly one candidate"
        )
    _validate_request(settings)

    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        structure = AseAtomsAdaptor.get_structure(parent)
        untrimmed = _build_grain_boundary(structure, settings, overlap_tolerance=0.0)
        constructed = _build_grain_boundary(
            structure,
            settings,
            overlap_tolerance=settings.grain_boundary_overlap_tolerance,
        )
        raw_atoms: Any = AseAtomsAdaptor.get_atoms(untrimmed)
        candidate: Any = AseAtomsAdaptor.get_atoms(constructed)
    except GrainBoundaryConstructionError:
        raise
    except Exception as exc:
        raise GrainBoundaryConstructionError(
            "failed to construct the supported Sigma-5 [001] (210) grain boundary: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    candidate.set_pbc((True, True, True))
    raw_atoms.set_pbc((True, True, True))
    thickness, normal_repeat = _grain_thickness(candidate, settings.grain_boundary_plane)
    if thickness + 1.0e-8 < settings.grain_boundary_min_thickness:
        raise GrainBoundaryConstructionError(
            "constructed grain thickness "
            f"{thickness:.8g} Å is below the configured minimum "
            f"{settings.grain_boundary_min_thickness:.8g} Å"
        )

    removed_species = _removed_species(raw_atoms, candidate)
    parent_distance = _minimum_parent_distance(structure)
    removal_distance = settings.grain_boundary_overlap_tolerance * parent_distance
    parameters = _parameters(
        candidate,
        constructed,
        base,
        settings,
        thickness=thickness,
        normal_repeat=normal_repeat,
        removed_species=removed_species,
        removal_distance=removal_distance,
    )
    parent_structure_id = calculate_structure_id(base)
    annotate(
        candidate,
        base,
        "grain_boundary",
        random_seed=derive_child_seed(
            parent_structure_id,
            settings.random_seed if seed is None else seed,
            "grain_boundary",
            0,
        ),
        parameters=parameters,
        operation_id="grain_boundary:sigma5_[001]_36.869897645844_(210)",
    )
    return [candidate]


def _validate_request(settings: PerturbationSettings) -> None:
    axis = tuple(settings.grain_boundary_rotation_axis)
    plane = tuple(settings.grain_boundary_plane)
    if axis != SUPPORTED_ROTATION_AXIS:
        raise GrainBoundaryConstructionError(
            "unsupported grain-boundary rotation axis; only (0, 0, 1) is supported"
        )
    if plane != SUPPORTED_BOUNDARY_PLANE:
        raise GrainBoundaryConstructionError(
            "unsupported grain-boundary plane; only (2, 1, 0) is supported"
        )
    if settings.grain_boundary_sigma != SUPPORTED_SIGMA:
        raise GrainBoundaryConstructionError(
            "unsupported grain-boundary Sigma; only Sigma 5 is supported"
        )
    if not math.isclose(
        settings.grain_boundary_misorientation_angle,
        SUPPORTED_MISORIENTATION_ANGLE,
        rel_tol=1.0e-9,
        abs_tol=1.0e-8,
    ):
        raise GrainBoundaryConstructionError(
            "unsupported grain-boundary misorientation angle; only 36.86989764584402 degrees is supported"
        )
    if settings.grain_boundary_expand_times <= 0:
        raise GrainBoundaryConstructionError("grain_boundary_expand_times must be positive")
    if settings.grain_boundary_min_thickness < 0.0:
        raise GrainBoundaryConstructionError("grain_boundary_min_thickness must be non-negative")
    if not 0.0 <= settings.grain_boundary_overlap_tolerance <= 1.0:
        raise GrainBoundaryConstructionError("grain_boundary_overlap_tolerance must be in [0, 1]")


def _build_grain_boundary(
    structure: Any,
    settings: PerturbationSettings,
    *,
    overlap_tolerance: float,
) -> Any:
    try:
        from pymatgen.core.interface import GrainBoundaryGenerator

        generator = GrainBoundaryGenerator(structure)
        axis = tuple(int(value) for value in settings.grain_boundary_rotation_axis)
        boundary_plane = tuple(int(value) for value in settings.grain_boundary_plane)
        grain_boundary = generator.gb_from_parameters(
            rotation_axis=(axis[0], axis[1], axis[2]),
            rotation_angle=float(settings.grain_boundary_misorientation_angle),
            expand_times=settings.grain_boundary_expand_times,
            plane=(boundary_plane[0], boundary_plane[1], boundary_plane[2]),
            rm_ratio=overlap_tolerance,
        )
    except Exception as exc:
        raise GrainBoundaryConstructionError(
            f"Pymatgen grain-boundary construction failed: {type(exc).__name__}: {exc}"
        ) from exc
    if int(grain_boundary.sigma) != SUPPORTED_SIGMA:
        raise GrainBoundaryConstructionError(
            "Pymatgen returned a grain boundary with an unexpected Sigma "
            f"{grain_boundary.sigma}; requested {SUPPORTED_SIGMA}"
        )
    return grain_boundary


def _removed_species(raw_atoms: Any, candidate: Any) -> dict[str, int]:
    raw_counts = Counter(raw_atoms.get_chemical_symbols())
    final_counts = Counter(candidate.get_chemical_symbols())
    removed = raw_counts - final_counts
    if sum(removed.values()) > len(raw_atoms) - len(candidate):
        raise GrainBoundaryConstructionError(
            "grain-boundary overlap accounting found an inconsistent species count"
        )
    return dict(sorted(removed.items()))


def _minimum_parent_distance(structure: Any) -> float:
    source = structure.copy()
    if len(source) == 1:
        source.make_supercell([1, 1, 2])
    distances = np.asarray(source.distance_matrix, dtype=float)
    nonzero = distances[distances > 1.0e-12]
    return float(np.min(nonzero)) if nonzero.size else 0.0


def _grain_thickness(
    candidate: Atoms,
    plane: tuple[int, int, int],
) -> tuple[float, float]:
    normal = np.asarray(plane, dtype=float)
    normal /= np.linalg.norm(normal)
    cell = np.asarray(candidate.cell, dtype=float)
    projections = np.abs(cell @ normal)
    repeat = float(np.max(projections))
    return repeat / 2.0, repeat


def _parameters(
    candidate: Atoms,
    grain_boundary: Any,
    base: Any,
    settings: PerturbationSettings,
    *,
    thickness: float,
    normal_repeat: float,
    removed_species: dict[str, int],
    removal_distance: float,
) -> dict[str, Any]:
    cell = np.asarray(candidate.cell, dtype=float)
    removed_count = sum(removed_species.values())
    axis = tuple(int(value) for value in grain_boundary.rotation_axis)
    plane = tuple(int(value) for value in grain_boundary.gb_plane)
    join_plane = tuple(int(value) for value in grain_boundary.join_plane)
    relationship = "sigma5_[001]_36.869897645844_(210)"
    return {
        "parent_structure_id": calculate_structure_id(base),
        "grain_boundary_relationship": relationship,
        "grain_boundary_rotation_axis": axis,
        "grain_boundary_misorientation_angle": float(grain_boundary.rotation_angle),
        "grain_boundary_sigma": int(grain_boundary.sigma),
        "grain_boundary_plane": plane,
        "grain_boundary_join_plane": join_plane,
        "grain_boundary_expand_times": int(settings.grain_boundary_expand_times),
        "grain_boundary_cell": tuple(tuple(float(value) for value in row) for row in cell),
        "grain_boundary_cell_lengths": tuple(float(np.linalg.norm(row)) for row in cell),
        "grain_boundary_pbc": (True, True, True),
        "grain_boundary_grain_thickness": thickness,
        "grain_boundary_normal_repeat": normal_repeat,
        "grain_boundary_overlap_tolerance": float(settings.grain_boundary_overlap_tolerance),
        "grain_boundary_overlap_ratio": float(settings.grain_boundary_overlap_tolerance),
        "grain_boundary_overlap_distance": removal_distance,
        "grain_boundary_overlap_removal_method": "pymatgen_merge_sites",
        "grain_boundary_removed_atom_count": removed_count,
        "grain_boundary_removed_species": removed_species,
        "grain_boundary_removed_atoms": removed_species,
        "overlap_removed_count": removed_count,
        "overlap_removed_species": removed_species,
        "overlap_tolerance": float(settings.grain_boundary_overlap_tolerance),
    }


__all__ = [
    "GrainBoundaryConstructionError",
    "SUPPORTED_BOUNDARY_PLANE",
    "SUPPORTED_MISORIENTATION_ANGLE",
    "SUPPORTED_ROTATION_AXIS",
    "SUPPORTED_SIGMA",
    "grain_boundaries",
]
