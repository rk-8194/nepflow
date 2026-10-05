"""Authoritative dataset assembly for NEP training.

This module owns the boundary between selected structures, verified DFT
results, and the persisted NEP dataset.  VASP interpretation deliberately
stays in :mod:`nepflow.dft.vasp.outputs`; this module only resolves the output
whose identity was already established by the canonical VASP APIs and records
the result in the dataset manifest.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import numpy as np

from nepflow.dft.vasp.outputs import (
    ResolvedVaspOutput,
    VaspParseResult,
    parse_outcar_result,
    validate_dft_result_labels,
)
from nepflow.domain.datasets import (
    DATASET_MANIFEST_SCHEMA,
    DatasetIdentity,
    SelectedDatasetMember,
    TrainingDatasetManifest,
)
from nepflow.domain.identities import StructureIdentity, calculate_structure_id
from nepflow.domain.units import (
    ENERGY_UNIT_EV,
    FORCE_UNIT_EV_PER_ANGSTROM,
    VIRIAL_CONVENTION_POSITIVE_COMPRESSION,
    VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3,
    VIRIAL_UNIT_EV,
    require_tensor_shape,
)
from nepflow.errors import StateError
from nepflow.io.hashing import sha256_file
from nepflow.io.json import read_json_object, to_jsonable, write_json
from nepflow.stages.selection.artifacts import read_selection_manifest

logger = logging.getLogger(__name__)


class DatasetSplit(str, Enum):
    """A persisted dataset split; it is explicit at every call boundary."""

    TRAIN = "train"
    TEST = "test"

    @classmethod
    def coerce(cls, value: "DatasetSplit | str") -> "DatasetSplit":
        """Normalize a split spelling or raise for an unsupported value."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported dataset split: {value!r}") from exc


@dataclass
class DatasetBuildReport:
    """Typed accounting for one explicit dataset split."""

    split: DatasetSplit
    requested_count: int = 0
    accepted_results: list[VaspParseResult] = field(default_factory=list)
    accepted_content_records: list[dict[str, Any]] = field(default_factory=list)
    rejected_reason_counts: dict[str, int] = field(default_factory=dict)
    requested_members: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        """Return the number of accepted typed or content-backed records."""
        return len(self.accepted_results) + len(self.accepted_content_records)

    @property
    def rejected_count(self) -> int:
        """Return requested minus accepted records for this split."""
        return self.requested_count - self.accepted_count

    def reset(self, requested_count: int) -> None:
        """Clear prior accounting and set the new requested count."""
        self.requested_count = int(requested_count)
        self.accepted_results.clear()
        self.accepted_content_records.clear()
        self.rejected_reason_counts.clear()
        self.requested_members.clear()

    def record_requested(self, structure_id: str | None, ordinal: int) -> int:
        """Record one requested member and return its report index."""
        self.requested_members.append(
            {
                "split": self.split.value,
                "ordinal": int(ordinal),
                "structure_id": structure_id,
                "status": "requested",
            }
        )
        return len(self.requested_members) - 1

    def record_rejection(self, reason: str, *, member_index: int | None = None) -> None:
        """Record a visible rejection reason and optionally update its member."""
        reason = str(reason)
        self.rejected_reason_counts[reason] = self.rejected_reason_counts.get(reason, 0) + 1
        if member_index is not None and member_index < len(self.requested_members):
            member = self.requested_members[member_index]
            member["status"] = "rejected"
            member["rejection_reason"] = reason

    def record_acceptance(
        self,
        result: VaspParseResult | None = None,
        *,
        content_record: Mapping[str, Any] | None = None,
        member_index: int | None = None,
    ) -> None:
        """Record an accepted parsed result or explicit content record."""
        if result is not None:
            self.accepted_results.append(result)
            identity = dict(result.calculation_identity)
            member = {
                "split": self.split.value,
                "structure_id": result.structure_id,
                "calculation_identity": identity,
                "source_outcar": result.source_outcar,
                "source_outcar_hash": result.source_outcar_hash,
                "status": "accepted",
            }
        elif content_record is not None:
            record = dict(to_jsonable(content_record))
            self.accepted_content_records.append(record)
            member = {
                "split": self.split.value,
                "structure_id": record.get("structure_id"),
                "status": "accepted",
                "record_type": record.get("record_type", "content"),
            }
        else:
            raise ValueError("An accepted result or content record is required")

        if member_index is not None and member_index < len(self.requested_members):
            self.requested_members[member_index].update(member)
        else:
            self.requested_members.append(member)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-shaped split accounting and provenance."""
        return {
            "split": self.split.value,
            "requested_count": self.requested_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "accepted_results": list(self.accepted_results),
            "accepted_content_records": to_jsonable(self.accepted_content_records),
            "rejected_reason_counts": dict(self.rejected_reason_counts),
            "requested_members": to_jsonable(self.requested_members),
        }


@dataclass(frozen=True)
class DatasetBuildResult:
    """Materialized dataset files plus their content-derived manifest."""

    dataset_path: Path
    manifest: TrainingDatasetManifest
    metadata: Mapping[str, Any]
    reports: Mapping[DatasetSplit, DatasetBuildReport]

    @property
    def dataset_id(self) -> str:
        """Return the content-derived dataset identity."""
        return self.manifest.identity.dataset_id

    @property
    def train_count(self) -> int:
        """Return the accepted training-record count."""
        return self.reports[DatasetSplit.TRAIN].accepted_count

    @property
    def test_count(self) -> int:
        """Return the accepted test-record count."""
        return self.reports[DatasetSplit.TEST].accepted_count


@dataclass(frozen=True)
class DatasetBuildPreparation:
    """Validated, content-derived dataset inputs before publication."""

    manifest: TrainingDatasetManifest
    metadata: Mapping[str, Any]
    reports: Mapping[DatasetSplit, DatasetBuildReport]
    rendered: Mapping[DatasetSplit, Sequence[Mapping[str, Any]]]


def _reset_or_create_report(
    split: DatasetSplit,
    requested_count: int,
    report: DatasetBuildReport | None,
) -> DatasetBuildReport:
    if report is None:
        report = DatasetBuildReport(split)
        report.reset(requested_count)
    elif report.split != split:
        raise ValueError("Dataset report split does not match the input split")
    elif report.requested_count == 0 and not report.requested_members:
        report.reset(requested_count)
    elif report.requested_count != requested_count:
        raise ValueError("Dataset report requested_count does not match input")
    return report


def _record_structure_id(value: Any) -> str | None:
    if isinstance(value, VaspParseResult):
        return value.structure_id
    if isinstance(value, Mapping):
        structure_id = value.get("structure_id")
        if structure_id is not None:
            return str(structure_id)
        nested = value.get("result", value.get("parse_result"))
        if nested is not None:
            return _record_structure_id(nested)
    return None


def _rejection(
    report: DatasetBuildReport,
    reason: str,
    *,
    member_index: int | None = None,
) -> None:
    logger.warning(
        "[%s] Rejecting requested dataset member%s: %s",
        report.split.value,
        "" if member_index is None else f" #{member_index}",
        reason,
    )
    report.record_rejection(reason, member_index=member_index)


def _result_from_mapping(value: Mapping[str, Any]) -> VaspParseResult:
    nested = value.get("result", value.get("parse_result"))
    if isinstance(nested, VaspParseResult):
        return nested
    source_identity = value.get("calculation_identity", {})
    if isinstance(source_identity, str):
        source_identity = {"calculation_id": source_identity}
    if not isinstance(source_identity, Mapping):
        raise ValueError("dataset result calculation_identity must be an object")

    accepted = value.get("status", "accepted") == "accepted"
    species = tuple(str(item) for item in value.get("species", ()))
    forces = value.get("forces_ev_per_angstrom", value.get("forces"))
    virial = value.get("virial_ev", value.get("virial"))
    positions = value.get("positions_angstrom", value.get("positions"))
    lattice = value.get("lattice_angstrom", value.get("lattice"))
    return VaspParseResult(
        structure_id=str(value["structure_id"]),
        calculation_identity=tuple(
            sorted((str(key), str(item)) for key, item in source_identity.items())
        ),
        source_outcar=str(value.get("source_outcar", value.get("outcar_path", ""))),
        source_outcar_hash=value.get("source_outcar_hash"),
        status="accepted" if accepted else "rejected",
        rejection_reason=value.get("rejection_reason"),
        energy_ev=value.get("energy_ev", value.get("energy")) if accepted else None,
        forces_ev_per_angstrom=forces if accepted else None,
        virial_ev=virial if accepted else None,
        positions_angstrom=positions if accepted else None,
        lattice_angstrom=lattice if accepted else None,
        species=species if accepted else (),
        pbc=tuple(bool(item) for item in value.get("pbc", ())) if accepted else (),
        energy_unit=str(value.get("energy_unit", ENERGY_UNIT_EV)),
        force_unit=str(value.get("force_unit", FORCE_UNIT_EV_PER_ANGSTROM)),
        virial_unit=str(value.get("virial_unit", VIRIAL_UNIT_EV)),
        virial_convention=str(
            value.get("virial_convention") or VIRIAL_CONVENTION_POSITIVE_COMPRESSION
        ),
        virial_tensor_convention=str(
            value.get("virial_tensor_convention") or VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3
        ),
    )


def _canonical_record(
    split: DatasetSplit,
    result: VaspParseResult,
    *,
    include_virial: bool,
) -> dict[str, Any]:
    if not result.accepted:
        raise ValueError("Only accepted VASP results can become dataset records")
    if result.virial_convention != VIRIAL_CONVENTION_POSITIVE_COMPRESSION:
        raise ValueError("Dataset virial must use the positive-compression convention")
    if result.virial_tensor_convention != VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3:
        raise ValueError("Dataset virial must use Cartesian 3x3 tensor ordering")
    forces = cast(np.ndarray, result.forces_ev_per_angstrom)
    positions = cast(np.ndarray, result.positions_angstrom)
    lattice = cast(np.ndarray, result.lattice_angstrom)
    record = {
        "split": split.value,
        "structure_id": result.structure_id,
        "calculation_identity": dict(result.calculation_identity),
        "source_outcar_hash": result.source_outcar_hash,
        "energy": result.energy_ev,
        "forces": forces.tolist(),
        "positions": positions.tolist(),
        "lattice": lattice.tolist(),
        "species": list(result.species),
        "pbc": list(result.pbc),
        "label_units": {
            "energy": result.energy_unit,
            "forces": result.force_unit,
        },
    }
    if include_virial:
        if result.virial_ev is None:
            raise ValueError("missing_required_virial")
        record["virial"] = result.virial_ev.tolist()
        record["label_units"]["virial"] = result.virial_unit
        record["virial_convention"] = result.virial_convention
    return record


def _normalise_explicit_item(value: Any) -> VaspParseResult | Mapping[str, Any]:
    if isinstance(value, (VaspParseResult, Mapping)):
        return value
    raise TypeError(
        "authoritative dataset members must be VaspParseResult records or result mappings; "
        "resolve ASE structures through the compatibility adapter first"
    )


def _build_report_metadata(
    dataset_path: Path,
    reports: Mapping[DatasetSplit, DatasetBuildReport],
    *,
    train_virial: bool,
    allow_partial: bool,
) -> tuple[dict[str, Any], TrainingDatasetManifest]:
    (
        canonical_records,
        accepted_identities,
        source_output_hashes,
        requested_members,
    ) = _collect_report_metadata_records(reports, train_virial=train_virial)

    accepted_count = sum(report.accepted_count for report in reports.values())
    if len(canonical_records) != accepted_count:
        raise RuntimeError("Accepted dataset records are missing immutable content provenance")

    label_schema, units = _dataset_label_schema(train_virial)
    identity_payload = {
        "schema_version": "nepflow.dataset.v1",
        "label_schema": label_schema,
        "units": units,
        "virial_convention": (VIRIAL_CONVENTION_POSITIVE_COMPRESSION if train_virial else None),
        "virial_tensor_convention": (
            VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3 if train_virial else None
        ),
        "records": canonical_records,
    }
    identity = DatasetIdentity.from_identity_payload(identity_payload)
    manifest = _build_report_manifest(identity, canonical_records, reports)
    train = reports[DatasetSplit.TRAIN]
    test = reports[DatasetSplit.TEST]
    train_reasons, test_reasons, rejection_reasons = _report_rejection_metadata(reports)

    created = manifest.created_at
    metadata: dict[str, Any] = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "dataset_id": identity.dataset_id,
        "dataset_schema_version": "nepflow.dataset.v1",
        "label_schema": label_schema,
        "created": created,
        "creation_timestamp": created,
        "dataset_folder": dataset_path.name,
        "requested_train_structures": train.requested_count,
        "requested_test_structures": test.requested_count,
        "accepted_train_structures": train.accepted_count,
        "accepted_test_structures": test.accepted_count,
        "rejected_train_structures": train.rejected_count,
        "rejected_test_structures": test.rejected_count,
        "train_structures": train.accepted_count,
        "test_structures": test.accepted_count,
        "total_structures": accepted_count,
        "rejection_reason_counts": rejection_reasons,
        "exclusion_reasons": rejection_reasons,
        "train_rejection_reason_counts": train_reasons,
        "test_rejection_reason_counts": test_reasons,
        "requested_members": requested_members,
        "accepted_members": accepted_identities,
        "accepted_calculation_identities": accepted_identities,
        "source_output_hashes": source_output_hashes,
        "records": canonical_records,
        "virial_required": train_virial,
        "virial_included": train_virial,
        "units": units,
        "virial_convention": VIRIAL_CONVENTION_POSITIVE_COMPRESSION,
        "virial_tensor_convention": VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3,
        "partial_dataset_allowed": allow_partial,
    }
    return metadata, manifest


def _collect_report_metadata_records(
    reports: Mapping[DatasetSplit, DatasetBuildReport],
    *,
    train_virial: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    canonical_records: list[dict[str, Any]] = []
    accepted_identities: list[dict[str, Any]] = []
    source_output_hashes: list[str] = []
    requested_members: list[dict[str, Any]] = []
    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        report = reports[split]
        requested_members.extend(report.requested_members)
        for result in report.accepted_results:
            canonical_records.append(_canonical_record(split, result, include_virial=train_virial))
            accepted_identities.append(
                {
                    "split": split.value,
                    "structure_id": result.structure_id,
                    "calculation_identity": dict(result.calculation_identity),
                    "source_outcar": result.source_outcar,
                    "source_outcar_hash": result.source_outcar_hash,
                }
            )
            if result.source_outcar_hash:
                source_output_hashes.append(result.source_outcar_hash)
        canonical_records.extend(report.accepted_content_records)
    return canonical_records, accepted_identities, source_output_hashes, requested_members


def _dataset_label_schema(train_virial: bool) -> tuple[dict[str, Any], dict[str, str]]:
    return (
        {
            "version": "nepflow.extxyz.labels.v1",
            "geometry": ["positions", "lattice", "species", "pbc"],
            "energy": True,
            "forces": True,
            "virial": train_virial,
        },
        {
            "energy": ENERGY_UNIT_EV,
            "forces": FORCE_UNIT_EV_PER_ANGSTROM,
            "virial": VIRIAL_UNIT_EV,
        },
    )


def _build_report_manifest(
    identity: DatasetIdentity,
    canonical_records: Sequence[Mapping[str, Any]],
    reports: Mapping[DatasetSplit, DatasetBuildReport],
) -> TrainingDatasetManifest:
    return TrainingDatasetManifest(
        identity=identity,
        records=tuple(canonical_records),
        selection_method="explicit_split",
        selection_parameters={
            "requested_train": reports[DatasetSplit.TRAIN].requested_count,
            "requested_test": reports[DatasetSplit.TEST].requested_count,
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _report_rejection_metadata(
    reports: Mapping[DatasetSplit, DatasetBuildReport],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    train_reasons = dict(reports[DatasetSplit.TRAIN].rejected_reason_counts)
    test_reasons = dict(reports[DatasetSplit.TEST].rejected_reason_counts)
    rejection_reasons: dict[str, int] = {}
    for reason, count in (*train_reasons.items(), *test_reasons.items()):
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + count
    return train_reasons, test_reasons, rejection_reasons


def _record_state_members(
    state_store: Any,
    manifest: TrainingDatasetManifest,
    reports: Mapping[DatasetSplit, DatasetBuildReport],
    *,
    project_id: str | None,
) -> None:
    state_store.upsert_dataset(manifest, project_id=project_id, status="prepared")
    ordinal = 0
    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        for result in reports[split].accepted_results:
            identity = dict(result.calculation_identity)
            calculation_id = identity.get("calculation_id")
            source_hash = result.source_outcar_hash
            if not calculation_id or not source_hash:
                raise ValueError(
                    "StateStore dataset members require calculation_id and source_outcar_hash"
                )
            state_store.upsert_structure(
                StructureIdentity(result.structure_id),
                metadata={"dataset_id": manifest.identity.dataset_id},
            )
            member = SelectedDatasetMember(
                split=split.value,
                structure_id=result.structure_id,
                calculation_id=calculation_id,
                source_outcar_hash=source_hash,
                ordinal=ordinal,
                calculation_identity=tuple(identity.items()),
            )
            state_store.record_dataset_member(
                manifest.identity.dataset_id,
                member,
                metadata={"source_outcar": result.source_outcar},
            )
            ordinal += 1


def _publish_dataset_artifacts(
    dataset_path: Path,
    rendered: Mapping[DatasetSplit, Sequence[Mapping[str, Any]]],
    metadata: Mapping[str, Any],
    *,
    train_virial: bool,
    expected_counts: Mapping[DatasetSplit, int],
) -> bool:
    """Materialize a complete dataset and atomically promote it.

    Returns whether an empty pre-existing target directory was replaced.  The
    caller uses that information to restore the target if StateStore
    finalization fails after publication.
    """

    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    target_was_empty = dataset_path.exists()
    if target_was_empty and (not dataset_path.is_dir() or any(dataset_path.iterdir())):
        raise FileExistsError(
            f"Dataset publication target is not an empty directory: {dataset_path}"
        )

    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{dataset_path.name}.staging-",
            dir=str(dataset_path.parent),
        )
    )
    published = False
    try:
        _stage_dataset_artifacts(
            staging_path,
            rendered,
            metadata,
            train_virial=train_virial,
            expected_counts=expected_counts,
        )
        _promote_staged_dataset(staging_path, dataset_path, target_was_empty)
        published = True
        return target_was_empty
    except BaseException:
        # Publication and rollback form one filesystem transaction; include
        # interruption so a cancelled build cannot leave a partial dataset.
        if published and dataset_path.exists():
            shutil.rmtree(dataset_path)
        if target_was_empty and not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise


def _stage_dataset_artifacts(
    staging_path: Path,
    rendered: Mapping[DatasetSplit, Sequence[Mapping[str, Any]]],
    metadata: Mapping[str, Any],
    *,
    train_virial: bool,
    expected_counts: Mapping[DatasetSplit, int],
) -> None:
    train_count = write_nep_dataset(
        staging_path / "train.xyz",
        rendered[DatasetSplit.TRAIN],
        include_virial=train_virial,
    )
    test_count = write_nep_dataset(
        staging_path / "test.xyz",
        rendered[DatasetSplit.TEST],
        include_virial=train_virial,
    )
    if train_count != expected_counts[DatasetSplit.TRAIN]:
        raise RuntimeError("Staged train dataset count does not match its report")
    if test_count != expected_counts[DatasetSplit.TEST]:
        raise RuntimeError("Staged test dataset count does not match its report")
    persisted_metadata = dict(to_jsonable(metadata))
    persisted_metadata["schema_version"] = DATASET_MANIFEST_SCHEMA
    persisted_metadata["artifacts"] = {
        filename: {
            "path": filename,
            "sha256": sha256_file(staging_path / filename, required=True),
        }
        for filename in ("train.xyz", "test.xyz")
    }
    write_json(staging_path / ".dataset", persisted_metadata)
    if isinstance(metadata, dict):
        metadata.clear()
        metadata.update(persisted_metadata)

    for filename in ("train.xyz", "test.xyz", ".dataset"):
        artifact = staging_path / filename
        if not artifact.is_file():
            raise RuntimeError(f"Staged dataset artifact is missing: {artifact}")


def _promote_staged_dataset(
    staging_path: Path,
    dataset_path: Path,
    target_was_empty: bool,
) -> None:
    if target_was_empty:
        dataset_path.rmdir()
    os.replace(staging_path, dataset_path)


def _rollback_published_dataset(dataset_path: Path, *, restore_empty_target: bool) -> None:
    """Remove only the dataset artifact published by this build."""

    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    if restore_empty_target:
        dataset_path.mkdir(parents=True, exist_ok=True)


def _require_state_authority(state_store: Any, result: VaspParseResult) -> None:
    """Require the parsed result and OUTCAR hash to exist in the ledger."""

    identity = dict(result.calculation_identity)
    calculation_id = identity.get("calculation_id")
    if not isinstance(calculation_id, str) or not calculation_id:
        raise ValueError("state_dft_calculation_id_missing")
    get_calculation = getattr(state_store, "get_dft_calculation", None)
    list_artifacts = getattr(state_store, "list_artifacts", None)
    if not callable(get_calculation) or not callable(list_artifacts):
        raise TypeError("authoritative dataset building requires StateStore DFT result APIs")
    calculation = get_calculation(calculation_id)
    if not isinstance(calculation, Mapping):
        raise ValueError("state_dft_calculation_missing")
    if calculation.get("status") != "completed":
        raise ValueError("state_dft_calculation_not_completed")
    if not calculation.get("accepted_attempt_id"):
        raise ValueError("state_dft_attempt_not_accepted")
    persisted_identity = calculation.get("identity", calculation.get("identity_json", {}))
    if isinstance(persisted_identity, Mapping):
        if dict(persisted_identity) != identity:
            raise ValueError("state_dft_identity_mismatch")
        if persisted_identity.get("structure_id") != result.structure_id:
            raise ValueError("state_dft_structure_identity_mismatch")
    raw_artifacts = list_artifacts(originating_attempt_id=str(calculation["accepted_attempt_id"]))
    if not isinstance(raw_artifacts, (list, tuple)):
        raise ValueError("state_dft_artifacts_malformed")
    artifacts = [item for item in raw_artifacts if isinstance(item, Mapping)]
    if not any(
        artifact.get("artifact_type") == "vasp_outcar"
        and artifact.get("sha256") == result.source_outcar_hash
        for artifact in artifacts
    ):
        raise ValueError("state_dft_outcar_hash_not_authoritative")


def build_training_dataset(
    dataset_path: Path,
    split_records: Mapping[DatasetSplit | str, Sequence[Any]],
    project_dir: Path,
    *,
    train_virial: bool = False,
    debug: bool = False,
    allow_partial: bool = False,
    state_store: Any | None = None,
    project_id: str | None = None,
    selection_method: str | None = None,
    selection_parameters: Mapping[str, Any] | None = None,
) -> DatasetBuildResult:
    """Build train/test files and a content-derived Phase 2-compatible manifest.

    ``split_records`` is deliberately explicit: callers cannot accidentally
    turn a boolean into a different split while moving data through the
    pipeline.  It accepts only verified :class:`VaspParseResult` records (or
    mappings containing one).  ASE structures must be resolved by the named
    compatibility adapter before this boundary.

    The dataset directory and manifest are published atomically, then accepted
    members are recorded in ``state_store``.  Rejected records are fatal unless
    ``allow_partial=True``; no synthetic scientific labels are created.
    """

    dataset_path = Path(dataset_path)
    preparation = prepare_training_dataset(
        split_records,
        dataset_path,
        train_virial=train_virial,
        allow_partial=allow_partial,
        state_store=state_store,
        selection_method=selection_method,
        selection_parameters=selection_parameters,
    )

    restore_empty_target = _publish_dataset_artifacts(
        dataset_path,
        preparation.rendered,
        preparation.metadata,
        train_virial=train_virial,
        expected_counts={
            DatasetSplit.TRAIN: preparation.reports[DatasetSplit.TRAIN].accepted_count,
            DatasetSplit.TEST: preparation.reports[DatasetSplit.TEST].accepted_count,
        },
    )
    try:
        _record_prepared_dataset(
            state_store,
            preparation,
            project_id=project_id,
        )
    except BaseException:
        # State publication is part of the dataset transaction.  Restore the
        # previous materialized dataset even when cancellation interrupts the
        # event write.
        _rollback_published_dataset(
            dataset_path,
            restore_empty_target=restore_empty_target,
        )
        raise
    return DatasetBuildResult(
        dataset_path,
        preparation.manifest,
        preparation.metadata,
        preparation.reports,
    )


def _record_prepared_dataset(
    state_store: Any,
    preparation: DatasetBuildPreparation,
    *,
    project_id: str | None,
) -> None:
    transaction = getattr(state_store, "transaction", None)
    if callable(transaction):
        transaction_context = cast(Callable[[], AbstractContextManager[Any]], transaction)
        with transaction_context():
            _record_state_members(
                state_store,
                preparation.manifest,
                preparation.reports,
                project_id=project_id,
            )
        return
    _record_state_members(
        state_store,
        preparation.manifest,
        preparation.reports,
        project_id=project_id,
    )


def prepare_training_dataset(
    split_records: Mapping[DatasetSplit | str, Sequence[Any]],
    dataset_path: Path,
    *,
    train_virial: bool = False,
    allow_partial: bool = False,
    state_store: Any | None = None,
    selection_method: str | None = None,
    selection_parameters: Mapping[str, Any] | None = None,
) -> DatasetBuildPreparation:
    """Resolve and validate exact labels without publishing a dataset.

    Training-stage orchestration uses this preview to derive the immutable
    dataset ID before deciding whether a materialization can be reused.

    ``state_store`` must be the authoritative DFT-result store.  Rendered
    records use eV, eV/Angstrom, Angstrom, and optionally positive-compression
    Cartesian virials in eV.
    """

    if state_store is None:
        raise ValueError("state_store is required for the authoritative training dataset path")

    normalized = _normalize_split_records(split_records)

    reports = {
        split: DatasetBuildReport(split, requested_count=len(normalized[split]))
        for split in (DatasetSplit.TRAIN, DatasetSplit.TEST)
    }
    rendered: dict[DatasetSplit, list[dict[str, Any]]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.TEST: [],
    }

    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        rendered[split] = _process_dataset_split(
            normalized[split],
            reports[split],
            state_store,
            train_virial=train_virial,
        )
    _validate_dataset_reports(reports, allow_partial=allow_partial)

    metadata, manifest = _build_report_metadata(
        Path(dataset_path),
        reports,
        train_virial=train_virial,
        allow_partial=allow_partial,
    )
    metadata, manifest = _apply_selection_metadata(
        metadata,
        manifest,
        selection_method=selection_method,
        selection_parameters=selection_parameters,
    )
    return DatasetBuildPreparation(manifest, metadata, reports, rendered)


def _normalize_split_records(
    split_records: Mapping[DatasetSplit | str, Sequence[Any]],
) -> dict[DatasetSplit, Sequence[Any]]:
    normalized: dict[DatasetSplit, Sequence[Any]] = {}
    for key, records in split_records.items():
        split = DatasetSplit.coerce(key)
        if split in normalized:
            raise ValueError(f"Duplicate dataset split: {split.value}")
        normalized[split] = tuple(_normalise_explicit_item(item) for item in records)
    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        normalized.setdefault(split, ())
    return normalized


def _process_dataset_split(
    records: Sequence[Any],
    report: DatasetBuildReport,
    state_store: Any,
    *,
    train_virial: bool,
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for ordinal, item in enumerate(records):
        member_index = report.record_requested(_record_structure_id(item), ordinal)
        try:
            result = item if isinstance(item, VaspParseResult) else _result_from_mapping(item)
            if not result.accepted:
                _rejection(
                    report,
                    result.rejection_reason or "record_rejected",
                    member_index=member_index,
                )
                continue
            _require_state_authority(state_store, result)
            validate_dft_result_labels(result.as_structure_dict())
            if not result.source_outcar_hash:
                _rejection(
                    report,
                    "missing_source_outcar_hash",
                    member_index=member_index,
                )
                continue
            if train_virial and result.virial_ev is None:
                _rejection(
                    report,
                    "missing_required_virial",
                    member_index=member_index,
                )
                continue
            report.record_acceptance(result, member_index=member_index)
            rendered.append(result.as_structure_dict())
        except Exception as exc:
            # One malformed selected result is recorded as a rejection with
            # its type.  It can only be omitted when the caller explicitly
            # enabled allow_partial; it is never turned into synthetic labels.
            _rejection(
                report,
                f"extraction_failed:{type(exc).__name__}",
                member_index=member_index,
            )
    return rendered


def _validate_dataset_reports(
    reports: Mapping[DatasetSplit, DatasetBuildReport],
    *,
    allow_partial: bool,
) -> None:
    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        if reports[split].rejected_count < 0:
            raise RuntimeError(f"Dataset report over-accepted {split.value} records")
    total_rejected = sum(report.rejected_count for report in reports.values())
    if total_rejected and not allow_partial:
        raise RuntimeError(
            "Dataset creation rejected selected structures; "
            "set train_nep.allow_partial_dataset=true to allow explicit partial data"
        )
    if (
        not reports[DatasetSplit.TRAIN].accepted_count
        or not reports[DatasetSplit.TEST].accepted_count
    ):
        raise RuntimeError("No valid structures found; cannot create a training dataset")


def _apply_selection_metadata(
    metadata: dict[str, Any],
    manifest: TrainingDatasetManifest,
    *,
    selection_method: str | None,
    selection_parameters: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], TrainingDatasetManifest]:
    if selection_method is not None:
        metadata["selection_method"] = selection_method
        manifest = TrainingDatasetManifest(
            identity=manifest.identity,
            records=manifest.records,
            selection_method=selection_method,
            selection_parameters=selection_parameters,
            created_at=manifest.created_at,
        )
    elif selection_parameters is not None:
        metadata["selection_parameters"] = to_jsonable(selection_parameters)
    return metadata, manifest


def resolve_selected_dft_results(
    project_dir: Path,
    state_store: Any,
    *,
    reader: Any | None = None,
) -> dict[DatasetSplit, tuple[VaspParseResult, ...]]:
    """Load selected structures and resolve their exact accepted DFT results.

    This canonical ``TrainingStage`` adapter requires manifest-bound accepted
    DFT results and rejects unverified filesystem-only records.
    """

    if reader is None:
        from ase.io import read as ase_read

        reader = ase_read
    by_structure = _index_selected_calculations(state_store)
    selection_manifest = read_selection_manifest(project_dir)

    resolved: dict[DatasetSplit, tuple[VaspParseResult, ...]] = {}
    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        source = Path(project_dir) / "structures" / "selected" / f"{split.value}.xyz"
        if not source.is_file():
            raise FileNotFoundError(
                f"Selected {split.value} structures not found at {source}; "
                "run the select stage first"
            )
        structures = reader(str(source), index=":", format="extxyz")
        if not isinstance(structures, list):
            structures = [structures]
        expected_ids = selection_manifest[f"{split.value}_structure_ids"]
        actual_ids = [calculate_structure_id(atoms) for atoms in structures]
        if actual_ids != expected_ids:
            raise StateError(
                f"Selected {split.value} structure content does not match its manifest"
            )
        resolved[split] = tuple(
            _resolve_selected_split(structures, split, by_structure, state_store)
        )
    return resolved


def _index_selected_calculations(state_store: Any) -> dict[str, list[Mapping[str, Any]]]:
    calculations = state_store.list_dft_calculations(
        statuses=("completed",),
        selected_only=True,
    )
    by_structure: dict[str, list[Mapping[str, Any]]] = {}
    for calculation in calculations:
        identity = calculation.get("identity", {})
        structure_id = identity.get("structure_id", calculation.get("structure_id"))
        if structure_id:
            by_structure.setdefault(str(structure_id), []).append(calculation)
    return by_structure


def _resolve_selected_split(
    structures: Sequence[Any],
    split: DatasetSplit,
    by_structure: Mapping[str, Sequence[Mapping[str, Any]]],
    state_store: Any,
) -> list[VaspParseResult]:
    results: list[VaspParseResult] = []
    for atoms in structures:
        structure_id = calculate_structure_id(atoms)
        requested_calculation_id = atoms.info.get("calculation_id")
        candidates = list(by_structure.get(structure_id, ()))
        if requested_calculation_id:
            candidates = [
                calculation
                for calculation in candidates
                if calculation.get("calculation_id") == requested_calculation_id
                or calculation.get("identity", {}).get("calculation_id") == requested_calculation_id
            ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"StateStore must resolve exactly one completed {split.value} DFT "
                f"calculation for structure_id={structure_id}; found {len(candidates)}"
            )
        results.append(
            _resolve_selected_calculation(
                atoms,
                structure_id,
                candidates[0],
                state_store,
            )
        )
    return results


def _resolve_selected_calculation(
    atoms: Any,
    structure_id: str,
    calculation: Mapping[str, Any],
    state_store: Any,
) -> VaspParseResult:
    identity = dict(calculation.get("identity", {}))
    calculation_id = str(calculation["calculation_id"])
    attempt_id = calculation.get("accepted_attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise RuntimeError(f"StateStore has no accepted attempt for {calculation_id}")
    artifacts = state_store.list_artifacts(originating_attempt_id=attempt_id)
    outcars = [artifact for artifact in artifacts if artifact.get("artifact_type") == "vasp_outcar"]
    if len(outcars) != 1:
        raise RuntimeError(f"StateStore has no unique accepted OUTCAR for {calculation_id}")
    outcar_path = Path(str(outcars[0].get("path", "")))
    if not outcar_path.is_file() or sha256_file(outcar_path) != outcars[0].get("sha256"):
        raise RuntimeError(f"Accepted OUTCAR artifact is missing or changed for {calculation_id}")
    evidence = ResolvedVaspOutput(
        outcar_path=outcar_path,
        calculation_identity=tuple(
            sorted((str(key), str(value)) for key, value in identity.items())
        ),
        verification_source="state_store_artifact",
    )
    result = parse_outcar_result(
        outcar_path,
        atoms,
        require_virial=False,
        calculation_identity=identity,
        identity_evidence=evidence,
    )
    if not result.accepted or result.structure_id != structure_id:
        raise RuntimeError(
            f"StateStore OUTCAR could not be accepted for {calculation_id}: "
            f"{result.rejection_reason or 'structure_identity_changed'}"
        )
    return result


def load_materialized_dataset(
    dataset_path: Path,
    state_store: Any,
) -> tuple[TrainingDatasetManifest, Mapping[str, Any]]:
    """Load a dataset only when artifacts and ledger identity both verify.

    File hashes, manifest schema, and the StateStore dataset identity are
    checked; changed or missing files raise rather than becoming empty data.
    """

    dataset_path = Path(dataset_path)
    metadata_path = dataset_path / ".dataset"
    if not dataset_path.is_dir() or not metadata_path.is_file():
        raise FileNotFoundError(f"Materialized training dataset is missing: {dataset_path}")
    metadata = read_json_object(metadata_path)
    if metadata.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise StateError(
            f"Unsupported materialized dataset manifest schema: {metadata.get('schema_version')!r}"
        )
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise StateError(f"Materialized dataset artifact index is malformed: {metadata_path}")
    for filename in ("train.xyz", "test.xyz"):
        artifact = artifacts.get(filename)
        if not isinstance(artifact, Mapping) or artifact.get("path") != filename:
            raise StateError(f"Materialized dataset artifact metadata is malformed: {filename}")
        artifact_path = dataset_path / filename
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get("sha256"):
            raise StateError(f"Materialized dataset artifact changed: {artifact_path}")
    dataset_id = metadata.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"Materialized dataset has no dataset_id: {metadata_path}")
    row = state_store.get_dataset(dataset_id)
    if row is None:
        raise StateError(f"Materialized dataset is not registered: {dataset_id}")
    stored = row.get("manifest", {})
    stored_identity = stored.get("identity", {}) if isinstance(stored, Mapping) else {}
    if not isinstance(stored_identity, Mapping):
        raise StateError(f"Dataset identity is malformed in StateStore: {dataset_id}")
    identity_payload = dict(stored_identity)
    identity_payload.pop("dataset_id", None)
    identity = DatasetIdentity(dataset_id, identity_payload)
    if identity.to_dict() != stored_identity:
        raise StateError(f"Dataset identity is inconsistent in StateStore: {dataset_id}")
    if metadata.get("dataset_id") != identity.dataset_id:
        raise StateError(f"Materialized dataset identity changed: {dataset_id}")
    manifest = TrainingDatasetManifest(
        identity=identity,
        records=tuple(stored.get("records", ())),
        selection_method=stored.get("selection_method"),
        selection_parameters=stored.get("selection_parameters"),
        descriptor_model_fingerprint=stored.get("descriptor_model_fingerprint"),
        created_at=stored.get("created_at"),
        code_version=stored.get("code_version"),
    )
    return manifest, metadata


def build_dataset_metadata(
    dataset_path: Path,
    train_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
    *,
    train_virial: bool,
    allow_partial: bool,
) -> dict[str, Any]:
    """Build the canonical manifest shape from split accounting records.

    This helper does not publish files or mutate StateStore.
    """

    reports: dict[DatasetSplit, DatasetBuildReport] = {}
    for split, source in (
        (DatasetSplit.TRAIN, train_report),
        (DatasetSplit.TEST, test_report),
    ):
        report = DatasetBuildReport(
            split,
            requested_count=int(source.get("requested_count", 0)),
        )
        report.accepted_results.extend(source.get("accepted_results", ()))
        report.accepted_content_records.extend(
            dict(item) for item in source.get("accepted_content_records", ())
        )
        report.rejected_reason_counts.update(
            {
                str(reason): int(count)
                for reason, count in source.get("rejected_reason_counts", {}).items()
            }
        )
        reports[split] = report
    metadata, _ = _build_report_metadata(
        Path(dataset_path),
        reports,
        train_virial=train_virial,
        allow_partial=allow_partial,
    )
    return metadata


def write_nep_dataset(
    output_path: Path,
    structures: Iterable[Mapping[str, Any]],
    *,
    include_virial: bool = False,
) -> int:
    """Write the canonical NEP extxyz label schema and return frame count.

    Positions and forces are ``(n_atoms, 3)`` arrays in Angstrom and
    eV/Angstrom respectively; lattice is a Cartesian ``(3, 3)`` matrix in
    Angstrom, energy is eV, and optional virial is a positive-compression
    Cartesian ``(3, 3)`` tensor in eV.  Species and atom rows retain their
    input order.  Invalid shapes, units, or non-finite labels raise
    ``ValueError``; output is written atomically as UTF-8.
    """

    rendered = [
        _render_nep_structure(structure, include_virial=include_virial) for structure in structures
    ]

    from nepflow.io.atomic import atomic_write_text

    atomic_write_text(output_path, "".join(rendered), encoding="utf-8")
    return len(rendered)


def _render_nep_structure(
    structure: Mapping[str, Any],
    *,
    include_virial: bool,
) -> str:
    species = list(structure["species"])
    positions = np.asarray(structure["positions"], dtype=float)
    forces = np.asarray(structure["forces"], dtype=float)
    lattice = np.asarray(structure["lattice"], dtype=float)
    energy = float(structure["energy"])
    if not np.isfinite(energy):
        raise ValueError("Dataset energy label is not finite")
    if positions.shape != (len(species), 3):
        raise ValueError("Dataset positions do not match species")
    if forces.shape != (len(species), 3) or not np.isfinite(forces).all():
        raise ValueError("Dataset forces are invalid")
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("Dataset lattice is invalid")
    if not np.isfinite(positions).all():
        raise ValueError("Dataset positions are not finite")
    pbc = list(structure["pbc"])
    if len(pbc) != 3:
        raise ValueError("Dataset pbc must contain three flags")
    virial = structure.get("virial")
    virial_array = _validated_virial(virial, include_virial)
    pbc_text = " ".join("T" if flag else "F" for flag in pbc)
    lattice_text = " ".join(f"{value:.10f}" for value in lattice.flat)
    header = (
        f'energy={energy:.10f} pbc="{pbc_text}" '
        f'Lattice="{lattice_text}" '
        "Properties=species:S:1:pos:R:3:force:R:3"
    )
    if virial_array is not None:
        virial_text = " ".join(f"{value:.10f}" for value in virial_array.flat)
        header += f' virial="{virial_text}"'
    lines = [f"{len(species)}\n{header}\n"]
    lines.extend(
        f"{symbol:3s} {position[0]:15.10f} {position[1]:15.10f} {position[2]:15.10f} "
        f"{force[0]:15.10f} {force[1]:15.10f} {force[2]:15.10f}\n"
        for symbol, position, force in zip(species, positions, forces)
    )
    return "".join(lines)


def _validated_virial(
    virial: Any,
    include_virial: bool,
) -> np.ndarray | None:
    if not include_virial:
        return None
    if virial is None:
        raise ValueError("missing_required_virial")
    try:
        virial_array = require_tensor_shape(virial, (3, 3), name="Dataset virial")
    except (TypeError, ValueError) as exc:
        raise ValueError("Dataset virial is invalid") from exc
    if not np.isfinite(virial_array).all():
        raise ValueError("Dataset virial is invalid")
    return virial_array


__all__ = [
    "DATASET_MANIFEST_SCHEMA",
    "DatasetBuildReport",
    "DatasetBuildPreparation",
    "DatasetBuildResult",
    "DatasetSplit",
    "build_dataset_metadata",
    "build_training_dataset",
    "load_materialized_dataset",
    "prepare_training_dataset",
    "resolve_selected_dft_results",
    "write_nep_dataset",
]
