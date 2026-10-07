"""Immutable identity primitives for Nepflow domain artifacts.

The identity payloads in this module are deliberately boring.  They are
versioned, deterministic, and keep operational metadata separate from the
scientific inputs that define an artifact's meaning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from nepflow.io.hashing import sha256_bytes, sha256_canonical_json
from nepflow.io.json import to_jsonable

STRUCTURE_IDENTITY_SCHEMA = "structure-v1"
CANDIDATE_IDENTITY_SCHEMA = "candidate-v1"
ARTIFACT_IDENTITY_SCHEMA = "nepflow.artifact_identity.v1"
DFT_CALCULATION_IDENTITY_SCHEMA = "nepflow.dft_calculation_identity.v1"
MODEL_RUN_IDENTITY_SCHEMA = "nepflow.model_run_identity.v1"
VALIDATION_RUN_IDENTITY_SCHEMA = "nepflow.validation_run_identity.v1"
DESCRIPTOR_CACHE_SCHEMA = "descriptor-cache-v2"


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(f"{float(values[index]):.14f}" for index in range(3))


def _pbc_values(atoms: Any) -> tuple[bool, bool, bool]:
    pbc = getattr(atoms, "pbc", (True, True, True))
    tolist = getattr(pbc, "tolist", None)
    if callable(tolist):
        pbc = tolist()
    if isinstance(pbc, bool):
        return (pbc, pbc, pbc)
    values = list(cast(Sequence[Any], pbc))
    if len(values) == 0:
        return (True, True, True)
    if len(values) == 1:
        value = bool(values[0])
        return value, value, value
    return cast(tuple[bool, bool, bool], tuple(bool(value) for value in values[:3]))


def canonical_structure_text(atoms: Any) -> str:
    """Return the versioned canonical text used for structure identity.

    Fractional coordinates are grouped by sorted element while preserving the
    original order within each element.  Labels and other mutable annotations
    are intentionally not part of this representation.  The cell is a
    Cartesian ``(3, 3)`` matrix and all three periodic-boundary flags are
    recorded explicitly.
    """

    symbols = list(atoms.get_chemical_symbols())
    unique_elements = sorted(set(symbols))
    scaled_positions = atoms.get_scaled_positions()
    cell = atoms.get_cell()
    pbc = _pbc_values(atoms)

    lines = [STRUCTURE_IDENTITY_SCHEMA]
    lines.append("pbc " + " ".join("1" if value else "0" for value in pbc))
    lines.append("cell")
    for row in cell:
        lines.append(_format_vector(row))
    lines.append("atoms")
    for element in unique_elements:
        for index, symbol in enumerate(symbols):
            if symbol == element:
                lines.append(f"{element} {_format_vector(scaled_positions[index])}")
    return "\n".join(lines) + "\n"


def calculate_structure_id(atoms: Any) -> str:
    """Hash the canonical structure bytes into a stable structure ID.

    The ID changes with cell, periodic flags, species, or fractional
    positions, but not with mutable ``Atoms.info`` annotations.
    """

    return sha256_bytes(canonical_structure_text(atoms).encode("utf-8"))


def annotate_structure_id(atoms: Any, *, overwrite: bool = True) -> str:
    """Attach and return the canonical ID in ``atoms.info``.

    ``overwrite=False`` preserves an existing annotation; the input object is
    otherwise mutated as a deliberate provenance side effect.
    """
    structure_id = calculate_structure_id(atoms)
    info = getattr(atoms, "info", None)
    if info is None:
        atoms.info = {}
        info = atoms.info
    if overwrite or "structure_id" not in info:
        info["structure_id"] = structure_id
        info["structure_id_version"] = STRUCTURE_IDENTITY_SCHEMA
    return str(info["structure_id"])


def annotate_structure_ids(structures: list[Any], *, overwrite: bool = True) -> None:
    """Annotate every structure in a generated batch."""

    for atoms in structures:
        annotate_structure_id(atoms, overwrite=overwrite)


def calculate_candidate_id(
    structure_id: str,
    magnetic_state_id: str | Any | None = None,
    *,
    magnetic_state: str | Any | None = None,
) -> str:
    """Return the stable candidate identity for one physical structure.

    A non-magnetic candidate deliberately reuses structure_id so existing
    selection, storage, and DFT consumers remain compatible. An explicit
    magnetic state receives a separate versioned identity while retaining the
    same physical structure ID.
    """

    if magnetic_state_id is not None and magnetic_state is not None:
        raise ValueError("provide only one of magnetic_state_id or magnetic_state")
    if magnetic_state_id is None:
        magnetic_state_id = magnetic_state
    structure = str(structure_id).strip()
    if not structure:
        raise ValueError("structure_id must not be blank")
    if magnetic_state_id is None:
        return structure
    state = getattr(magnetic_state_id, "magnetic_state_id", magnetic_state_id)
    state_id = str(state).strip()
    if not state_id:
        raise ValueError("magnetic_state_id must not be blank when supplied")
    payload = {
        "schema_version": CANDIDATE_IDENTITY_SCHEMA,
        "structure_id": structure,
        "magnetic_state_id": state_id,
    }
    return "candidate_" + sha256_canonical_json(payload)


def calculate_magnetic_state_id(
    moments: Sequence[Sequence[Any]],
    constraint_mask: Sequence[Any],
    **kwargs: Any,
) -> str:
    """Compatibility entry point for the cross-stage magnetic identity."""

    from .magnetism import calculate_magnetic_state_id as _calculate_magnetic_state_id

    return _calculate_magnetic_state_id(moments, constraint_mask, **kwargs)


def annotate_candidate_id(
    atoms: Any,
    *,
    magnetic_state_id: str | Any | None = None,
    overwrite: bool = True,
) -> str:
    """Attach physical and candidate IDs without changing structure identity."""

    structure_id = calculate_structure_id(atoms)
    state = magnetic_state_id
    if state is None:
        state = getattr(atoms, "info", {}).get("magnetic_state_id")
    candidate_id = calculate_candidate_id(structure_id, state)
    info = getattr(atoms, "info", None)
    if info is None:
        atoms.info = {}
        info = atoms.info
    if overwrite or "structure_id" not in info:
        info["structure_id"] = structure_id
        info["structure_id_version"] = STRUCTURE_IDENTITY_SCHEMA
    if overwrite or "candidate_id" not in info:
        info["candidate_id"] = candidate_id
        info["candidate_id_version"] = CANDIDATE_IDENTITY_SCHEMA
    if state is not None and (overwrite or "magnetic_state_id" not in info):
        info["magnetic_state_id"] = str(getattr(state, "magnetic_state_id", state))
    return str(info["candidate_id"])


@dataclass(frozen=True)
class CandidateIdentity:
    """Versioned identity joining physical and explicit magnetic identity."""

    structure_id: str
    magnetic_state_id: str | None = None
    schema_version: str = CANDIDATE_IDENTITY_SCHEMA
    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_IDENTITY_SCHEMA:
            raise ValueError(
                f"CandidateIdentity schema_version must be {CANDIDATE_IDENTITY_SCHEMA!r}"
            )
        structure = str(self.structure_id).strip()
        if not structure:
            raise ValueError("structure_id must not be blank")
        state = None if self.magnetic_state_id is None else str(self.magnetic_state_id).strip()
        if state == "":
            raise ValueError("magnetic_state_id must not be blank when supplied")
        object.__setattr__(self, "structure_id", structure)
        object.__setattr__(self, "magnetic_state_id", state)
        object.__setattr__(
            self,
            "candidate_id",
            calculate_candidate_id(structure, state),
        )

    @classmethod
    def from_magnetic_state(
        cls,
        structure_id: str,
        magnetic_state: Any,
    ) -> "CandidateIdentity":
        return cls(
            structure_id=structure_id,
            magnetic_state_id=getattr(magnetic_state, "magnetic_state_id", magnetic_state),
        )

    @property
    def is_magnetic(self) -> bool:
        return self.magnetic_state_id is not None

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "structure_id": self.structure_id,
            "magnetic_state_id": self.magnetic_state_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            **self.identity_payload(),
        }


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-shaped metadata for immutable records."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if hasattr(value, "tolist") and callable(value.tolist):
        return _freeze(value.tolist())
    return value


def _normalise_pairs(
    values: Mapping[str, Any] | Sequence[tuple[str, Any]],
) -> tuple[tuple[str, Any], ...]:
    if isinstance(values, Mapping):
        items = values.items()
    else:
        items = values
    return tuple(
        sorted(((str(key), to_jsonable(value)) for key, value in items), key=lambda item: item[0])
    )


def normalise_dft_calculation_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a persisted DFT identity without changing its scientific inputs."""

    result = dict(value)
    if "structure_id" not in result and isinstance(result.get("structure_hash"), str):
        result["structure_id"] = result["structure_hash"]
    if "calculation_id" not in result and all(
        isinstance(result.get(key), str) and result[key].strip()
        for key in ("structure_id", "incar_hash", "potcar_hash")
    ):
        result["calculation_id"] = DftCalculationIdentity(
            structure_id=result["structure_id"],
            incar_hash=result["incar_hash"],
            potcar_hash=result["potcar_hash"],
        ).calculation_id
    result.pop("structure_hash", None)
    return result


@dataclass(frozen=True)
class StructureIdentity:
    """Versioned identity of one canonical structure."""

    structure_id: str
    schema_version: str = STRUCTURE_IDENTITY_SCHEMA

    @classmethod
    def from_atoms(cls, atoms: Any) -> "StructureIdentity":
        """Create an identity from an ASE-like structure."""
        return cls(calculate_structure_id(atoms))

    def to_dict(self) -> dict[str, str]:
        """Serialize the structure ID and schema version."""
        return {
            "structure_id": self.structure_id,
            "structure_id_version": self.schema_version,
        }


@dataclass(frozen=True)
class ArtifactIdentity:
    """Content hash and provenance for one persisted artifact."""

    artifact_id: str
    artifact_type: str
    sha256: str
    path: str | None = None
    schema_version: str = ARTIFACT_IDENTITY_SCHEMA

    @classmethod
    def from_bytes(
        cls, artifact_type: str, content: bytes, *, path: str | None = None
    ) -> "ArtifactIdentity":
        """Create an artifact identity from exact bytes and optional path."""
        digest = sha256_bytes(content)
        artifact_id = "artifact_" + sha256_canonical_json(
            {
                "schema_version": ARTIFACT_IDENTITY_SCHEMA,
                "artifact_type": artifact_type,
                "sha256": digest,
            }
        )
        return cls(artifact_id, artifact_type, digest, path)

    @classmethod
    def from_file(cls, artifact_type: str, path: str | Path) -> "ArtifactIdentity":
        """Hash a file into an artifact identity without guessing on I/O failure."""
        resolved = Path(path).resolve()
        return cls.from_bytes(artifact_type, resolved.read_bytes(), path=str(resolved))

    def to_dict(self) -> dict[str, str]:
        """Serialize the artifact type, digest, schema, and optional path."""
        result = {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
        }
        if self.path is not None:
            result["path"] = self.path
        return result


@dataclass(frozen=True)
class DftCalculationIdentity:
    """Identity of scientific DFT inputs, separate from execution resources.

    Resource fields are retained in the historical identity payload for
    compatibility.  In particular, ``gpus`` means GPUs per node; callers can
    use :attr:`gpus_per_node` to make that scope explicit.
    """

    structure_id: str
    incar_hash: str
    potcar_hash: str
    poscar_hash: str | None = None
    kpoints_hash: str | None = None
    backend_inputs_hash: str | None = None
    executable_version: str | None = None
    nodes: str | None = None
    ncore: int | None = None
    kpar: int | None = None
    gpus: int | None = None
    walltime: str | None = None
    calculation_id: str = field(init=False)

    @property
    def gpus_per_node(self) -> int | None:
        """Return the legacy GPU field with its node scope made explicit."""

        return self.gpus

    def __post_init__(self) -> None:
        payload = self.scientific_payload()
        object.__setattr__(self, "calculation_id", "calculation_" + sha256_canonical_json(payload))

    def scientific_payload(self) -> dict[str, Any]:
        """Return hash-defining DFT inputs, excluding execution resources."""
        payload: dict[str, Any] = {
            "schema_version": DFT_CALCULATION_IDENTITY_SCHEMA,
            "structure_id": self.structure_id,
            "incar_hash": self.incar_hash,
            "potcar_hash": self.potcar_hash,
        }
        for key, value in (
            ("poscar_hash", self.poscar_hash),
            ("kpoints_hash", self.kpoints_hash),
            ("backend_inputs_hash", self.backend_inputs_hash),
            ("executable_version", self.executable_version),
        ):
            if value is not None:
                payload[key] = value
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Serialize scientific identity and explicitly retained resources."""
        result = dict(self.scientific_payload())
        result["calculation_id"] = self.calculation_id
        resources = {
            key: value
            for key, value in (
                ("nodes", self.nodes),
                ("ncore", self.ncore),
                ("kpar", self.kpar),
                ("gpus", self.gpus),
                ("walltime", self.walltime),
            )
            if value is not None
        }
        if resources:
            result["resources"] = resources
        return result


@dataclass(frozen=True)
class ModelRunIdentity:
    """Identity of one dataset/input/hyperparameter training run."""

    dataset_id: str
    nep_in_sha256: str
    hyperparameters_hash: str
    model_run_id: str = field(init=False)
    schema_version: str = MODEL_RUN_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_run_id", "model_run_" + sha256_canonical_json(self.identity_payload())
        )

    @classmethod
    def from_inputs(
        cls, dataset_id: str, nep_in_sha256: str, hyperparameters_hash: str
    ) -> "ModelRunIdentity":
        return cls(dataset_id, nep_in_sha256, hyperparameters_hash)

    def identity_payload(self) -> dict[str, str]:
        """Return the ordered, hash-defining model training inputs."""
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "nep_in_sha256": self.nep_in_sha256,
            "hyperparameters_hash": self.hyperparameters_hash,
        }

    def to_dict(self) -> dict[str, str]:
        """Serialize this model-run identity for state and manifests."""
        return {"model_run_id": self.model_run_id, **self.identity_payload()}


@dataclass(frozen=True)
class ValidationRunIdentity:
    """Identity of one model/dataset/settings validation campaign."""

    model_run_id: str
    dataset_id: str
    validation_settings: Mapping[str, Any] = field(default_factory=dict)
    validation_run_id: str = field(init=False)
    schema_version: str = VALIDATION_RUN_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        settings = dict(_normalise_pairs(self.validation_settings))
        object.__setattr__(self, "validation_settings", _freeze(settings))
        object.__setattr__(
            self,
            "validation_run_id",
            "validation_run_" + sha256_canonical_json(self.identity_payload()),
        )

    def identity_payload(self) -> dict[str, Any]:
        """Return normalized validation inputs used to derive the run ID."""
        return {
            "schema_version": self.schema_version,
            "model_run_id": self.model_run_id,
            "dataset_id": self.dataset_id,
            "validation_settings": self.validation_settings,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the validation-run identity and its settings."""
        return {"validation_run_id": self.validation_run_id, **to_jsonable(self.identity_payload())}


@dataclass(frozen=True)
class DescriptorCacheIdentity:
    """Identity of an ordered descriptor cache and its model fingerprint."""

    structure_ids: tuple[str, ...]
    model_filename: str
    model_sha256: str
    mean_descriptor: bool
    descriptor_shape: tuple[int, ...]
    schema_version: str = DESCRIPTOR_CACHE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "structure_ids", tuple(self.structure_ids))
        object.__setattr__(
            self, "descriptor_shape", tuple(int(value) for value in self.descriptor_shape)
        )

    def to_manifest(self) -> dict[str, Any]:
        """Return the cache manifest, including ordered IDs and model hash."""
        return {
            "schema_version": self.schema_version,
            "structure_ids": list(self.structure_ids),
            "model": {"filename": self.model_filename, "sha256": self.model_sha256},
            "settings": {"mean_descriptor": self.mean_descriptor},
            "descriptor_shape": list(self.descriptor_shape),
        }

    @property
    def cache_id(self) -> str:
        """Return the deterministic ID for this exact descriptor cache."""
        return "descriptor_cache_" + sha256_canonical_json(self.to_manifest())
