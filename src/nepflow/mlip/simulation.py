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

    def __post_init__(self) -> None:
        if self.atom_count < 1:
            raise ValidationError("static prediction atom_count must be positive")
        if self.model.artifact is None:
            raise ValidationError(
                "static prediction requires a model run with a model artifact"
            )


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
