"""Typed validation case and reference-label records.

Validation used to pass nested dictionaries between preparation, execution,
and analysis.  That made it possible for a case to lose the model/dataset
association or for geometry metadata to disappear before GPUMD was called.
These records are the small, explicit boundary shared by those components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nepflow.domain.identities import ModelRunIdentity, StructureIdentity
from nepflow.domain.models import ModelRunRecord
from nepflow.io.hashing import sha256_canonical_json
from nepflow.io.json import to_jsonable
from nepflow.mlip.simulation import StaticPredictionRequest
from nepflow.errors import ValidationError


VALIDATION_CASE_SCHEMA = "nepflow.validation_case.v1"
VALIDATION_PREPARATION_SCHEMA = "nepflow.validation_preparation.v1"


def _immutable_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} is not numeric") from exc
    if array.shape != shape:
        raise ValidationError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValidationError(f"{label} contains non-finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _pbc(value: Any) -> tuple[bool, bool, bool]:
    if isinstance(value, bool):
        return (value, value, value)
    values = tuple(bool(item) for item in value)
    if len(values) != 3:
        raise ValidationError("validation case pbc must contain exactly three flags")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ValidationReference:
    """Authoritative DFT labels and geometry for one validation structure."""

    structure: StructureIdentity
    species: tuple[str, ...]
    positions_angstrom: np.ndarray
    cell_angstrom: np.ndarray
    pbc: tuple[bool, bool, bool]
    energy_ev: float
    forces_ev_per_angstrom: np.ndarray
    virial_ev: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        species = tuple(str(value) for value in self.species)
        if not species:
            raise ValidationError("validation reference must contain at least one atom")
        object.__setattr__(self, "species", species)
        object.__setattr__(
            self,
            "positions_angstrom",
            _immutable_array(
                self.positions_angstrom,
                (len(species), 3),
                "validation reference positions",
            ),
        )
        object.__setattr__(
            self,
            "cell_angstrom",
            _immutable_array(self.cell_angstrom, (3, 3), "validation reference cell"),
        )
        object.__setattr__(self, "pbc", _pbc(self.pbc))
        try:
            energy = float(self.energy_ev)
        except (TypeError, ValueError) as exc:
            raise ValidationError("validation reference energy is not numeric") from exc
        if not np.isfinite(energy):
            raise ValidationError("validation reference energy is not finite")
        object.__setattr__(self, "energy_ev", energy)
        object.__setattr__(
            self,
            "forces_ev_per_angstrom",
            _immutable_array(
                self.forces_ev_per_angstrom,
                (len(species), 3),
                "validation reference forces",
            ),
        )
        if self.virial_ev is not None:
            object.__setattr__(
                self,
                "virial_ev",
                _immutable_array(self.virial_ev, (3, 3), "validation reference virial"),
            )
        if self.metadata is not None:
            object.__setattr__(self, "metadata", to_jsonable(dict(self.metadata)))

    @property
    def structure_id(self) -> str:
        return self.structure.structure_id

    @property
    def atom_count(self) -> int:
        return len(self.species)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "structure": self.structure.to_dict(),
            "species": list(self.species),
            "positions_angstrom": self.positions_angstrom.tolist(),
            "cell_angstrom": self.cell_angstrom.tolist(),
            "pbc": list(self.pbc),
            "energy_ev": self.energy_ev,
            "forces_ev_per_angstrom": self.forces_ev_per_angstrom.tolist(),
            "virial_ev": None if self.virial_ev is None else self.virial_ev.tolist(),
        }
        if self.metadata is not None:
            result["metadata"] = to_jsonable(self.metadata)
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ValidationReference":
        structure_value = value.get("structure", {})
        if isinstance(structure_value, Mapping):
            structure_id = structure_value.get("structure_id")
        else:
            structure_id = value.get("structure_id")
        if not isinstance(structure_id, str) or not structure_id.strip():
            raise ValidationError("persisted validation reference is missing structure_id")
        return cls(
            structure=StructureIdentity(
                structure_id,
                str(
                    structure_value.get("structure_id_version", "structure-v1")
                    if isinstance(structure_value, Mapping)
                    else "structure-v1"
                ),
            ),
            species=tuple(value["species"]),
            positions_angstrom=value["positions_angstrom"],
            cell_angstrom=value["cell_angstrom"],
            pbc=tuple(value["pbc"]),
            energy_ev=value["energy_ev"],
            forces_ev_per_angstrom=value["forces_ev_per_angstrom"],
            virial_ev=value.get("virial_ev"),
            metadata=value.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class ValidationCaseSpec:
    """One immutable model/dataset-bound GPUMD validation case."""

    case_id: str
    ordinal: int
    model_run_id: str
    dataset_id: str
    reference: ValidationReference
    input_path: Path
    working_directory: Path
    output_path: Path
    replicates: tuple[int, int, int] = (1, 1, 1)
    virial_requested: bool = False
    schema_version: str = VALIDATION_CASE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("case_id", "model_run_id", "dataset_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValidationError(f"validation case {name} must not be blank")
            object.__setattr__(self, name, value)
        if self.ordinal < 0:
            raise ValidationError("validation case ordinal cannot be negative")
        replicates = tuple(int(value) for value in self.replicates)
        if len(replicates) != 3 or any(value < 1 for value in replicates):
            raise ValidationError("validation case replicates must be three positive integers")
        object.__setattr__(self, "replicates", replicates)
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "working_directory", Path(self.working_directory))
        object.__setattr__(self, "output_path", Path(self.output_path))

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        model_run_id: str,
        dataset_id: str,
        reference: ValidationReference,
        input_path: Path,
        working_directory: Path,
        output_path: Path,
        replicates: tuple[int, int, int] = (1, 1, 1),
        virial_requested: bool = False,
    ) -> "ValidationCaseSpec":
        payload = {
            "schema_version": VALIDATION_CASE_SCHEMA,
            "ordinal": int(ordinal),
            "model_run_id": str(model_run_id),
            "dataset_id": str(dataset_id),
            "structure_id": reference.structure_id,
        }
        case_id = "validation_case_" + sha256_canonical_json(payload)
        return cls(
            case_id=case_id,
            ordinal=ordinal,
            model_run_id=model_run_id,
            dataset_id=dataset_id,
            reference=reference,
            input_path=input_path,
            working_directory=working_directory,
            output_path=output_path,
            replicates=replicates,
            virial_requested=virial_requested,
        )

    @property
    def structure_id(self) -> str:
        return self.reference.structure_id

    @property
    def atom_count(self) -> int:
        return self.reference.atom_count

    @property
    def cell_angstrom(self) -> np.ndarray:
        return self.reference.cell_angstrom

    @property
    def pbc(self) -> tuple[bool, bool, bool]:
        return self.reference.pbc

    def static_prediction_request(self, model: ModelRunRecord) -> StaticPredictionRequest:
        """Create the existing MLIP static-prediction request for this case."""

        if model.model_run_id != self.model_run_id:
            raise ValidationError(
                "validation case model_run_id does not match the supplied model"
            )
        if model.identity.dataset_id != self.dataset_id:
            raise ValidationError(
                "validation case dataset_id does not match the supplied model"
            )
        return StaticPredictionRequest(
            structure=self.reference.structure,
            model=model,
            input_path=self.input_path,
            working_directory=self.working_directory,
            atom_count=self.atom_count * int(np.prod(self.replicates)),
            virial_requested=self.virial_requested,
            species=self.reference.species,
            positions_angstrom=self.reference.positions_angstrom,
            cell_angstrom=self.reference.cell_angstrom,
            pbc=self.reference.pbc,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            "model_run_id": self.model_run_id,
            "dataset_id": self.dataset_id,
            "structure_id": self.structure_id,
            "input_path": str(self.input_path),
            "working_directory": str(self.working_directory),
            "output_path": str(self.output_path),
            "replicates": list(self.replicates),
            "virial_requested": self.virial_requested,
            "reference": self.reference.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ValidationCaseSpec":
        try:
            reference = ValidationReference.from_mapping(value["reference"])
            return cls(
                case_id=str(value["case_id"]),
                ordinal=int(value["ordinal"]),
                model_run_id=str(value["model_run_id"]),
                dataset_id=str(value["dataset_id"]),
                reference=reference,
                input_path=Path(str(value["input_path"])),
                working_directory=Path(str(value["working_directory"])),
                output_path=Path(str(value["output_path"])),
                replicates=tuple(int(item) for item in value.get("replicates", (1, 1, 1))),
                virial_requested=bool(value.get("virial_requested", False)),
                schema_version=str(value.get("schema_version", VALIDATION_CASE_SCHEMA)),
            )
        except KeyError as exc:
            raise ValidationError(f"persisted validation case is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True, slots=True)
class ValidationPreparation:
    """Resolved model/dataset pair and its deterministic case matrix."""

    model_run_id: str
    dataset_id: str
    model_path: Path
    dataset_path: Path
    cases: tuple[ValidationCaseSpec, ...]
    schema_version: str = VALIDATION_PREPARATION_SCHEMA

    def __post_init__(self) -> None:
        if not str(self.model_run_id).strip() or not str(self.dataset_id).strip():
            raise ValidationError("validation preparation requires model and dataset IDs")
        object.__setattr__(self, "model_path", Path(self.model_path))
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        cases = tuple(self.cases)
        if tuple(case.ordinal for case in cases) != tuple(range(len(cases))):
            raise ValidationError("validation cases must have contiguous deterministic ordinals")
        for case in cases:
            if case.model_run_id != self.model_run_id or case.dataset_id != self.dataset_id:
                raise ValidationError("validation case identity does not match preparation")
        object.__setattr__(self, "cases", cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_run_id": self.model_run_id,
            "dataset_id": self.dataset_id,
            "model_path": str(self.model_path),
            "dataset_path": str(self.dataset_path),
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ValidationPreparation":
        try:
            cases = tuple(
                ValidationCaseSpec.from_mapping(item)
                for item in value["cases"]
            )
            return cls(
                model_run_id=str(value["model_run_id"]),
                dataset_id=str(value["dataset_id"]),
                model_path=Path(str(value["model_path"])),
                dataset_path=Path(str(value["dataset_path"])),
                cases=cases,
                schema_version=str(
                    value.get("schema_version", VALIDATION_PREPARATION_SCHEMA)
                ),
            )
        except KeyError as exc:
            raise ValidationError(
                f"persisted validation preparation is missing {exc.args[0]!r}"
            ) from exc


# ``ValidationCase`` is the concise public spelling used by callers that do
# not need to distinguish the immutable specification from runtime status.
ValidationCase = ValidationCaseSpec


__all__ = [
    "VALIDATION_CASE_SCHEMA",
    "VALIDATION_PREPARATION_SCHEMA",
    "ValidationCaseSpec",
    "ValidationCase",
    "ValidationPreparation",
    "ValidationReference",
]
