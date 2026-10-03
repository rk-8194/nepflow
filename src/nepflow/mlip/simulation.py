"""Narrow static-prediction boundary used by current GPUMD validation.

The current validation consumer needs one genuine model evaluation per
structure.  This module intentionally does not model property suites, MD
trajectories, or active-learning exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from nepflow.domain.identities import ModelRunIdentity, StructureIdentity
from nepflow.domain.models import ModelRunRecord
from nepflow.domain.units import (
    ENERGY_UNIT_EV,
    FORCE_UNIT_EV_PER_ANGSTROM,
    VIRIAL_CONVENTION_POSITIVE_COMPRESSION,
    VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3,
    VIRIAL_UNIT_EV,
)
from nepflow.errors import ValidationError


@dataclass(frozen=True, slots=True)
class StaticPredictionRequest:
    """One model/structure request for a genuine static prediction."""

    structure: StructureIdentity
    model: ModelRunRecord
    input_path: Path
    working_directory: Path
    atom_count: int
    virial_requested: bool = False
    species: tuple[str, ...] | None = None
    positions_angstrom: np.ndarray | None = None
    cell_angstrom: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] | None = None
    expected_species: tuple[str, ...] | None = None
    expected_positions_angstrom: np.ndarray | None = None
    expected_cell_angstrom: np.ndarray | None = None
    expected_pbc: tuple[bool, bool, bool] | None = None
    atom_mapping: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.atom_count < 1:
            raise ValidationError("static prediction atom_count must be positive")
        if self.model.artifact is None:
            raise ValidationError(
                "static prediction requires a model run with a model artifact"
            )
        expected_fields = (
            self.expected_species,
            self.expected_positions_angstrom,
            self.expected_cell_angstrom,
            self.expected_pbc,
        )
        expected_present = tuple(value is not None for value in expected_fields)
        if any(expected_present) and not all(expected_present):
            raise ValidationError(
                "static prediction expected configuration must provide species, "
                "positions, cell, and pbc together"
            )
        if self.atom_mapping is not None and not all(expected_present):
            raise ValidationError(
                "static prediction atom_mapping requires complete expected configuration"
            )
        if self.species is not None:
            species = tuple(str(value) for value in self.species)
            if not species:
                raise ValidationError("static prediction species must not be empty")
            object.__setattr__(self, "species", species)
        if self.positions_angstrom is not None:
            positions = np.asarray(self.positions_angstrom, dtype=float)
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValidationError(
                    "static prediction reference positions must have shape (n_atoms, 3)"
                )
            if not np.isfinite(positions).all():
                raise ValidationError("static prediction reference positions are not finite")
            object.__setattr__(self, "positions_angstrom", np.array(positions, copy=True))
        if self.cell_angstrom is not None:
            cell = np.asarray(self.cell_angstrom, dtype=float)
            if cell.shape != (3, 3) or not np.isfinite(cell).all():
                raise ValidationError(
                    "static prediction reference cell must be a finite 3x3 matrix"
                )
            object.__setattr__(self, "cell_angstrom", np.array(cell, copy=True))
        if self.pbc is not None:
            values = tuple(bool(value) for value in self.pbc)
            if len(values) != 3:
                raise ValidationError("static prediction reference pbc must have three flags")
            object.__setattr__(self, "pbc", values)  # type: ignore[assignment]
        if self.expected_species is not None:
            expected_species = tuple(str(value) for value in self.expected_species)
            if len(expected_species) != self.atom_count:
                raise ValidationError(
                    "static prediction expected species count does not match atom_count"
                )
            object.__setattr__(self, "expected_species", expected_species)
        if self.expected_positions_angstrom is not None:
            expected_positions = np.asarray(self.expected_positions_angstrom, dtype=float)
            if expected_positions.shape != (self.atom_count, 3):
                raise ValidationError(
                    "static prediction expected positions must have shape "
                    f"({self.atom_count}, 3)"
                )
            if not np.isfinite(expected_positions).all():
                raise ValidationError(
                    "static prediction expected positions are not finite"
                )
            object.__setattr__(
                self, "expected_positions_angstrom", np.array(expected_positions, copy=True)
            )
        if self.expected_cell_angstrom is not None:
            expected_cell = np.asarray(self.expected_cell_angstrom, dtype=float)
            if expected_cell.shape != (3, 3) or not np.isfinite(expected_cell).all():
                raise ValidationError(
                    "static prediction expected cell must be a finite 3x3 matrix"
                )
            object.__setattr__(
                self, "expected_cell_angstrom", np.array(expected_cell, copy=True)
            )
        if self.expected_pbc is not None:
            expected_pbc = tuple(bool(value) for value in self.expected_pbc)
            if len(expected_pbc) != 3:
                raise ValidationError(
                    "static prediction expected pbc must have three flags"
                )
            object.__setattr__(self, "expected_pbc", expected_pbc)  # type: ignore[assignment]
        if self.atom_mapping is not None:
            mapping = tuple(int(value) for value in self.atom_mapping)
            if len(mapping) != self.atom_count or any(value < 0 for value in mapping):
                raise ValidationError(
                    "static prediction atom_mapping must match atom_count and use non-negative indices"
                )
            object.__setattr__(self, "atom_mapping", mapping)

    @property
    def cell(self) -> np.ndarray | None:
        """Compatibility spelling for the reference cell metadata."""

        return self.cell_angstrom

    @property
    def positions(self) -> np.ndarray | None:
        """Compatibility spelling for reference Cartesian positions."""

        return self.positions_angstrom


@dataclass(frozen=True, slots=True)
class PredictionRuntimeMetadata:
    """Execution metadata kept separate from scientific prediction values."""

    elapsed_seconds: float | None = None
    backend_version: str | None = None
    command: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class StaticPrediction:
    """Canonical static energy/force/virial values returned by a backend."""

    structure: StructureIdentity
    model_run: ModelRunIdentity
    atom_count: int
    energy_ev: float
    forces_ev_per_angstrom: np.ndarray
    virial_ev: np.ndarray | None = None
    virial_requested: bool = False
    runtime: PredictionRuntimeMetadata = PredictionRuntimeMetadata()
    energy_unit: str = ENERGY_UNIT_EV
    force_unit: str = FORCE_UNIT_EV_PER_ANGSTROM
    virial_unit: str = VIRIAL_UNIT_EV
    virial_convention: str = VIRIAL_CONVENTION_POSITIVE_COMPRESSION
    virial_tensor_convention: str = VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3
    species: tuple[str, ...] | None = None
    positions_angstrom: np.ndarray | None = None
    cell_angstrom: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] | None = None
    atom_mapping: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.atom_count < 1:
            raise ValidationError("static prediction atom_count must be positive")

        try:
            energy = float(self.energy_ev)
        except (TypeError, ValueError) as exc:
            raise ValidationError("static prediction is missing a numeric energy") from exc
        if not np.isfinite(energy):
            raise ValidationError("static prediction energy is not finite")
        object.__setattr__(self, "energy_ev", energy)

        forces = np.asarray(self.forces_ev_per_angstrom, dtype=float)
        if forces.shape != (self.atom_count, 3):
            raise ValidationError(
                "static prediction forces must have shape "
                f"({self.atom_count}, 3), got {forces.shape}"
            )
        if not np.isfinite(forces).all():
            raise ValidationError("static prediction forces contain non-finite values")
        forces = np.array(forces, copy=True)
        forces.setflags(write=False)
        object.__setattr__(self, "forces_ev_per_angstrom", forces)

        if self.virial_requested and self.virial_ev is None:
            raise ValidationError("static prediction is missing requested virial/stress")
        if self.virial_ev is not None:
            virial = np.asarray(self.virial_ev, dtype=float)
            if virial.shape != (3, 3):
                raise ValidationError(
                    "static prediction virial must have shape (3, 3), "
                    f"got {virial.shape}"
                )
            if not np.isfinite(virial).all():
                raise ValidationError(
                    "static prediction virial contains non-finite values"
                )
            virial = np.array(virial, copy=True)
            virial.setflags(write=False)
            object.__setattr__(self, "virial_ev", virial)

        if self.species is not None:
            species = tuple(str(value) for value in self.species)
            if len(species) != self.atom_count:
                raise ValidationError(
                    "static prediction species count does not match atom_count"
                )
            object.__setattr__(self, "species", species)
        if self.positions_angstrom is not None:
            positions = np.asarray(self.positions_angstrom, dtype=float)
            if positions.shape != (self.atom_count, 3):
                raise ValidationError(
                    "static prediction positions must have shape "
                    f"({self.atom_count}, 3), got {positions.shape}"
                )
            if not np.isfinite(positions).all():
                raise ValidationError("static prediction positions contain non-finite values")
            positions = np.array(positions, copy=True)
            positions.setflags(write=False)
            object.__setattr__(self, "positions_angstrom", positions)
        if self.cell_angstrom is not None:
            cell = np.asarray(self.cell_angstrom, dtype=float)
            if cell.shape != (3, 3) or not np.isfinite(cell).all():
                raise ValidationError(
                    "static prediction cell must be a finite 3x3 matrix"
                )
            cell = np.array(cell, copy=True)
            cell.setflags(write=False)
            object.__setattr__(self, "cell_angstrom", cell)
        if self.pbc is not None:
            values = tuple(bool(value) for value in self.pbc)
            if len(values) != 3:
                raise ValidationError("static prediction pbc must have three flags")
            object.__setattr__(self, "pbc", values)  # type: ignore[assignment]
        if self.atom_mapping is not None:
            mapping = tuple(int(value) for value in self.atom_mapping)
            if len(mapping) != self.atom_count or any(value < 0 for value in mapping):
                raise ValidationError(
                    "static prediction atom_mapping must match atom_count and use non-negative indices"
                )
            object.__setattr__(self, "atom_mapping", mapping)

    @property
    def cell(self) -> np.ndarray | None:
        """Compatibility spelling for the output cell metadata."""

        return self.cell_angstrom

    @property
    def positions(self) -> np.ndarray | None:
        """Compatibility spelling for output Cartesian positions."""

        return self.positions_angstrom


@runtime_checkable
class StaticPredictionBackend(Protocol):
    """Request one genuine static evaluation from an MLIP/simulation backend."""

    def predict(self, request: StaticPredictionRequest) -> StaticPrediction:
        """Return energy, forces, and requested virial with runtime metadata."""


__all__ = [
    "PredictionRuntimeMetadata",
    "StaticPrediction",
    "StaticPredictionBackend",
    "StaticPredictionRequest",
]
