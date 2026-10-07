"""Read-only access to the parent topology carried by generated structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms

from nepflow.stages.generation.supercell import (
    PARENT_TOPOLOGY_SCHEMA,
    require_parent_topology,
)


class MagneticTopologyError(ValueError):
    """Raised when a magnetic candidate lacks trustworthy parent topology."""


@dataclass(frozen=True, slots=True)
class ParentTopology:
    """Validated parent mapping copied from one structural candidate."""

    site_indices: np.ndarray
    orbit_indices: np.ndarray
    translations: np.ndarray
    unwrapped_fractional: np.ndarray
    mapped: np.ndarray
    transformation: np.ndarray
    repeat: tuple[int, int, int]
    symmetry_tolerance: float
    schema: str

    def __post_init__(self) -> None:
        arrays = (
            ("site_indices", self.site_indices, (len(self.site_indices),)),
            ("orbit_indices", self.orbit_indices, (len(self.orbit_indices),)),
            ("translations", self.translations, (len(self.site_indices), 3)),
            ("unwrapped_fractional", self.unwrapped_fractional, (len(self.site_indices), 3)),
            ("mapped", self.mapped, (len(self.site_indices),)),
        )
        for name, values, shape in arrays:
            normalized = np.asarray(values)
            if normalized.shape != shape:
                raise MagneticTopologyError(f"parent topology {name} has invalid shape")
            normalized = np.array(normalized, copy=True)
            normalized.setflags(write=False)
            object.__setattr__(self, name, normalized)
        matrix = np.asarray(self.transformation, dtype=int)
        if matrix.shape != (3, 3):
            raise MagneticTopologyError("parent topology transformation must be 3x3")
        matrix = np.array(matrix, copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "transformation", matrix)
        raw_repeat = tuple(int(value) for value in self.repeat)
        if len(raw_repeat) != 3 or any(value <= 0 for value in raw_repeat):
            raise MagneticTopologyError("parent topology repeat must contain positive integers")
        repeat = (raw_repeat[0], raw_repeat[1], raw_repeat[2])
        object.__setattr__(self, "repeat", repeat)
        if not np.isfinite(float(self.symmetry_tolerance)) or self.symmetry_tolerance <= 0.0:
            raise MagneticTopologyError("parent topology symmetry tolerance is invalid")
        object.__setattr__(self, "symmetry_tolerance", float(self.symmetry_tolerance))
        if self.schema != PARENT_TOPOLOGY_SCHEMA:
            raise MagneticTopologyError(f"unsupported parent topology schema: {self.schema!r}")

    @property
    def mapped_magnetic_orbits(self) -> tuple[int, ...]:
        return tuple(sorted(set(int(value) for value in self.orbit_indices[self.mapped])))

    def metadata(self) -> dict[str, Any]:
        return {
            "parent_topology_schema": self.schema,
            "topology_supercell_repeat": list(self.repeat),
            "topology_transformation": self.transformation.tolist(),
            "topology_symmetry_tolerance": self.symmetry_tolerance,
        }


def read_parent_topology(atoms: Atoms) -> ParentTopology:
    """Validate and read existing topology without attempting symmetry discovery."""

    try:
        require_parent_topology(atoms)
    except Exception as exc:
        if isinstance(exc, MagneticTopologyError):
            raise
        raise MagneticTopologyError(str(exc)) from exc
    info = atoms.info
    try:
        transformation = np.asarray(info["topology_transformation"], dtype=int)
        raw_repeat = tuple(int(value) for value in info["topology_supercell_repeat"])
        if len(raw_repeat) != 3:
            raise MagneticTopologyError("parent topology repeat must contain three integers")
        repeat = (raw_repeat[0], raw_repeat[1], raw_repeat[2])
        symmetry_tolerance = float(info["topology_symmetry_tolerance"])
        schema = str(info["parent_topology_schema"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise MagneticTopologyError("parent topology metadata is invalid") from exc
    return ParentTopology(
        site_indices=np.asarray(atoms.arrays["parent_site_index"], dtype=int),
        orbit_indices=np.asarray(atoms.arrays["parent_orbit_index"], dtype=int),
        translations=np.asarray(atoms.arrays["parent_cell_translation"], dtype=int),
        unwrapped_fractional=np.asarray(atoms.arrays["parent_unwrapped_fractional"], dtype=float),
        mapped=np.asarray(atoms.arrays["parent_topology_mapped"], dtype=bool),
        transformation=transformation,
        repeat=repeat,
        symmetry_tolerance=symmetry_tolerance,
        schema=schema,
    )


__all__ = ["MagneticTopologyError", "ParentTopology", "read_parent_topology"]
