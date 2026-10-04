"""Typed boundary for density-functional-theory backends.

The protocol deliberately describes the scientific/backend boundary only.  A
backend can prepare files and interpret its own output, while process
execution and scheduler orchestration remain owned by the corresponding
infrastructure and workflow layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from nepflow.domain.calculations import DftResultArtifact
from nepflow.domain.identities import DftCalculationIdentity, StructureIdentity
from nepflow.domain.units import (
    ENERGY_UNIT_EV,
    FORCE_UNIT_EV_PER_ANGSTROM,
    VIRIAL_CONVENTION_POSITIVE_COMPRESSION,
    VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3,
    VIRIAL_UNIT_EV,
)
from nepflow.errors import BackendError, ValidationError
from nepflow.hpc.process import ProcessResult


@dataclass(frozen=True, slots=True)
class DftResultRequirements:
    """Explicit required quantities for one parsed DFT result."""

    energy_required: bool = True
    forces_required: bool = True
    virial_requested: bool = False


@dataclass(frozen=True, slots=True)
class DftInputRequest:
    """Typed inputs needed to prepare one scientific DFT calculation."""

    structure: StructureIdentity
    source_structure: Path
    working_directory: Path
    requirements: DftResultRequirements = field(default_factory=DftResultRequirements)
    source_structure_index: int = 0


@dataclass(frozen=True, slots=True)
class DftInputArtifacts:
    """Prepared files and their canonical scientific calculation identity."""

    calculation: DftCalculationIdentity
    working_directory: Path
    files: tuple[Path, ...]
    requirements: DftResultRequirements = field(default_factory=DftResultRequirements)


@dataclass(frozen=True, slots=True)
class DftCompletionEvidence:
    """Backend completion evidence separated from scheduler state."""

    completed: bool
    markers: tuple[str, ...] = ()
    returncode: int | None = None


@dataclass(frozen=True, slots=True)
class DftResult:
    """Canonical result labels parsed from a completed DFT calculation.

    Energy is in eV, forces are Cartesian eV/Angstrom, and virial is an
    optional 3x3 tensor in eV using the positive-compression convention.
    The explicit requirements record determines which quantities may be
    absent; it is carried from the input request through the prepared inputs.
    """

    structure: StructureIdentity
    calculation: DftCalculationIdentity
    energy_ev: float | None
    forces_ev_per_angstrom: np.ndarray | None
    virial_ev: np.ndarray | None = None
    artifact: DftResultArtifact | None = None
    requirements: DftResultRequirements = field(default_factory=DftResultRequirements)
    energy_unit: str = ENERGY_UNIT_EV
    force_unit: str = FORCE_UNIT_EV_PER_ANGSTROM
    virial_unit: str = VIRIAL_UNIT_EV
    virial_convention: str = VIRIAL_CONVENTION_POSITIVE_COMPRESSION
    virial_tensor_convention: str = VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3

    def __post_init__(self) -> None:
        if self.energy_ev is None:
            if self.requirements.energy_required:
                raise BackendError("DFT result is missing required energy")
        else:
            try:
                energy = float(self.energy_ev)
            except (TypeError, ValueError) as exc:
                raise BackendError("DFT result energy is not numeric") from exc
            if not np.isfinite(energy):
                raise BackendError("DFT result energy is not finite")
            object.__setattr__(self, "energy_ev", energy)

        if self.calculation.structure_id != self.structure.structure_id:
            raise ValidationError(
                "DFT result structure identity does not match calculation identity"
            )

        if self.forces_ev_per_angstrom is None:
            if self.requirements.forces_required:
                raise BackendError("DFT result is missing required forces")
        else:
            forces = np.asarray(self.forces_ev_per_angstrom, dtype=float)
            if forces.ndim != 2 or forces.shape[1] != 3:
                raise BackendError(
                    f"DFT result forces must have shape (n_atoms, 3), got {forces.shape}"
                )
            if not np.isfinite(forces).all():
                raise BackendError("DFT result forces contain non-finite values")
            forces = np.array(forces, copy=True)
            forces.setflags(write=False)
            object.__setattr__(self, "forces_ev_per_angstrom", forces)

        if self.virial_ev is None:
            if self.requirements.virial_requested:
                raise BackendError("DFT result is missing requested virial/stress")
        else:
            virial = np.asarray(self.virial_ev, dtype=float)
            if virial.shape != (3, 3):
                raise BackendError(f"DFT result virial must have shape (3, 3), got {virial.shape}")
            if not np.isfinite(virial).all():
                raise BackendError("DFT result virial contains non-finite values")
            virial = np.array(virial, copy=True)
            virial.setflags(write=False)
            object.__setattr__(self, "virial_ev", virial)

        if (
            self.artifact is not None
            and self.artifact.calculation.calculation_id != self.calculation.calculation_id
        ):
            raise ValidationError("DFT result artifact identity does not match the parsed result")


@dataclass(frozen=True, slots=True)
class DftFailureEvidence:
    """Process/output evidence used by a DFT backend failure classifier."""

    calculation: DftCalculationIdentity
    completion: DftCompletionEvidence
    process: ProcessResult | None = None
    markers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DftFailure:
    """Typed backend failure classification for workflow policy."""

    kind: str
    recoverable: bool
    reason: str


@runtime_checkable
class DftBackend(Protocol):
    """Scientific boundary required by the current VASP workflow."""

    def prepare_inputs(self, request: DftInputRequest) -> DftInputArtifacts:
        """Render POSCAR/POTCAR/INCAR-like scientific input artifacts."""

    def execution_command(self, inputs: DftInputArtifacts) -> tuple[str, ...]:
        """Return an argument-oriented command; never an implicit shell body."""

    def parse_completion(
        self,
        inputs: DftInputArtifacts,
        process: ProcessResult | None = None,
    ) -> DftCompletionEvidence:
        """Determine completion from backend evidence, not scheduler absence."""

    def parse_result(self, inputs: DftInputArtifacts) -> DftResult:
        """Parse required scientific labels and canonicalize their units."""

    def classify_failure(self, evidence: DftFailureEvidence) -> DftFailure:
        """Classify backend output for workflow recovery/quarantine policy."""

    def calculation_identity(self, request: DftInputRequest) -> DftCalculationIdentity:
        """Expose the identity of the scientific inputs being prepared."""

    def validate_calculation_identity(
        self,
        expected: DftCalculationIdentity,
        observed: DftCalculationIdentity,
    ) -> None:
        """Reject output whose scientific identity does not match the request."""


__all__ = [
    "DftBackend",
    "DftCompletionEvidence",
    "DftFailure",
    "DftFailureEvidence",
    "DftInputArtifacts",
    "DftInputRequest",
    "DftResult",
    "DftResultRequirements",
]
