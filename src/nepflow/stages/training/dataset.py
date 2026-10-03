"""Authoritative dataset assembly for NEP training.

This module owns the boundary between selected structures, verified DFT
results, and the persisted NEP dataset.  VASP interpretation deliberately
stays in :mod:`nepflow.dft.vasp.outputs`; this module only resolves the output
whose identity was already established by the canonical VASP APIs and records
the result in the dataset manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

# Only used by the compatibility bridge below; the authoritative builder does
# not import or call an ASE reader.
ase_read = None

from nepflow.dft.vasp.outputs import VaspParseResult, validate_dft_result_labels
from nepflow.domain.datasets import (
    DatasetIdentity,
    SelectedDatasetMember,
    TrainingDatasetManifest,
)
from nepflow.domain.identities import StructureIdentity
from nepflow.domain.units import (
    ENERGY_UNIT_EV,
    FORCE_UNIT_EV_PER_ANGSTROM,
    VIRIAL_CONVENTION_POSITIVE_COMPRESSION,
    VIRIAL_UNIT_EV,
    VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3,
)
from nepflow.io.json import to_jsonable, write_json


logger = logging.getLogger("nepflow.training.dataset")


class DatasetSplit(str, Enum):
    """A persisted dataset split; it is explicit at every call boundary."""

    TRAIN = "train"
    TEST = "test"

    @classmethod
    def coerce(cls, value: "DatasetSplit | str") -> "DatasetSplit":
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
        return len(self.accepted_results) + len(self.accepted_content_records)

    @property
    def rejected_count(self) -> int:
        return self.requested_count - self.accepted_count

    def reset(self, requested_count: int) -> None:
        self.requested_count = int(requested_count)
        self.accepted_results.clear()
        self.accepted_content_records.clear()
        self.rejected_reason_counts.clear()
        self.requested_members.clear()

    def record_requested(self, structure_id: str | None, ordinal: int) -> int:
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
        reason = str(reason)
        self.rejected_reason_counts[reason] = (
            self.rejected_reason_counts.get(reason, 0) + 1
        )
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
        return self.manifest.identity.dataset_id

    @property
    def train_count(self) -> int:
        return self.reports[DatasetSplit.TRAIN].accepted_count

    @property
    def test_count(self) -> int:
        return self.reports[DatasetSplit.TEST].accepted_count


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


def iter_labeled_structures(*args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
    """Compatibility bridge to the legacy Atoms-to-VASP resolver.

    The authoritative dataset builder below does not call this function. It
    remains importable for pre-Phase 4 callers while the resolver itself lives
    in the explicitly named compatibility adapter.
    """

    from modules.train_nep.prepare import resolve_legacy_labeled_structures

    if "reader" not in kwargs:
        kwargs["reader"] = ase_read
    return resolve_legacy_labeled_structures(*args, **kwargs)


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
            value.get("virial_convention", VIRIAL_CONVENTION_POSITIVE_COMPRESSION)
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
    record = {
        "split": split.value,
        "structure_id": result.structure_id,
        "calculation_identity": dict(result.calculation_identity),
        "source_outcar_hash": result.source_outcar_hash,
        "energy": result.energy_ev,
        "forces": result.forces_ev_per_angstrom.tolist(),
        "positions": result.positions_angstrom.tolist(),
        "lattice": result.lattice_angstrom.tolist(),
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
    canonical_records: list[dict[str, Any]] = []
    accepted_identities: list[dict[str, Any]] = []
    source_output_hashes: list[str] = []
    requested_members: list[dict[str, Any]] = []
    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        report = reports[split]
        requested_members.extend(report.requested_members)
        for result in report.accepted_results:
            canonical_records.append(
                _canonical_record(split, result, include_virial=train_virial)
            )
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

    accepted_count = sum(report.accepted_count for report in reports.values())
    if len(canonical_records) != accepted_count:
        raise RuntimeError("Accepted dataset records are missing immutable content provenance")

    label_schema = {
        "version": "nepflow.extxyz.labels.v1",
        "geometry": ["positions", "lattice", "species", "pbc"],
        "energy": True,
        "forces": True,
        "virial": train_virial,
    }
    units = {
        "energy": ENERGY_UNIT_EV,
        "forces": FORCE_UNIT_EV_PER_ANGSTROM,
        "virial": VIRIAL_UNIT_EV,
    }
    identity_payload = {
        "schema_version": "nepflow.dataset.v1",
        "label_schema": label_schema,
        "units": units,
        "virial_convention": (
            VIRIAL_CONVENTION_POSITIVE_COMPRESSION if train_virial else None
        ),
        "virial_tensor_convention": (
            VIRIAL_TENSOR_CONVENTION_CARTESIAN_3X3 if train_virial else None
        ),
        "records": canonical_records,
    }
    identity = DatasetIdentity.from_identity_payload(identity_payload)
    manifest = TrainingDatasetManifest(
        identity=identity,
        records=tuple(canonical_records),
        selection_method="explicit_split",
        selection_parameters={
            "requested_train": reports[DatasetSplit.TRAIN].requested_count,
            "requested_test": reports[DatasetSplit.TEST].requested_count,
        },
        created_at=datetime.now().isoformat(),
    )
    train = reports[DatasetSplit.TRAIN]
    test = reports[DatasetSplit.TEST]
    train_reasons = dict(train.rejected_reason_counts)
    test_reasons = dict(test.rejected_reason_counts)
    rejection_reasons: dict[str, int] = {}
    for reason, count in (*train_reasons.items(), *test_reasons.items()):
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + count

    created = manifest.created_at
    metadata: dict[str, Any] = {
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


def _require_state_authority(state_store: Any, result: VaspParseResult) -> None:
    """Require the parsed result and OUTCAR hash to exist in the ledger."""

    identity = dict(result.calculation_identity)
    calculation_id = identity.get("calculation_id")
    if not isinstance(calculation_id, str) or not calculation_id:
        raise ValueError("state_dft_calculation_id_missing")
    get_calculation = getattr(state_store, "get_dft_calculation", None)
    list_artifacts = getattr(state_store, "list_artifacts", None)
    if not callable(get_calculation) or not callable(list_artifacts):
        raise TypeError(
            "authoritative dataset building requires StateStore DFT result APIs"
        )
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
    artifacts = list_artifacts(
        originating_attempt_id=str(calculation["accepted_attempt_id"])
    )
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
    """

    if state_store is None:
        raise ValueError(
            "state_store is required for the authoritative training dataset path"
        )

    normalized: dict[DatasetSplit, Sequence[Any]] = {}
    for key, records in split_records.items():
        split = DatasetSplit.coerce(key)
        if split in normalized:
            raise ValueError(f"Duplicate dataset split: {split.value}")
        normalized[split] = tuple(_normalise_explicit_item(item) for item in records)
    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        normalized.setdefault(split, ())

    reports = {
        split: DatasetBuildReport(split, requested_count=len(normalized[split]))
        for split in (DatasetSplit.TRAIN, DatasetSplit.TEST)
    }
    rendered: dict[DatasetSplit, list[dict[str, Any]]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.TEST: [],
    }

    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        report = reports[split]
        for ordinal, item in enumerate(normalized[split]):
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
                rendered[split].append(result.as_structure_dict())
            except Exception as exc:
                _rejection(
                    report,
                    f"extraction_failed:{type(exc).__name__}",
                    member_index=member_index,
                )

    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        if reports[split].rejected_count < 0:
            raise RuntimeError(f"Dataset report over-accepted {split.value} records")

    total_rejected = sum(report.rejected_count for report in reports.values())
    if total_rejected and not allow_partial:
        raise RuntimeError(
            "Dataset creation rejected selected structures; "
            "set train_nep.allow_partial_dataset=true to allow explicit partial data"
        )
    if not reports[DatasetSplit.TRAIN].accepted_count or not reports[DatasetSplit.TEST].accepted_count:
        raise RuntimeError("No valid structures found; cannot create a training dataset")

    # All acceptance and exact-count policy checks happen before any apparent
    # dataset artifact is published.  A rejected exact-count build therefore
    # cannot leave train.xyz/test.xyz behind without a completed manifest.
    dataset_path = Path(dataset_path)
    metadata, manifest = _build_report_metadata(
        dataset_path,
        reports,
        train_virial=train_virial,
        allow_partial=allow_partial,
    )
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

    _record_state_members(
        state_store,
        manifest,
        reports,
        project_id=project_id,
    )
    dataset_path.mkdir(parents=True, exist_ok=True)
    write_nep_dataset(
        dataset_path / "train.xyz",
        rendered[DatasetSplit.TRAIN],
        include_virial=train_virial,
    )
    write_nep_dataset(
        dataset_path / "test.xyz",
        rendered[DatasetSplit.TEST],
        include_virial=train_virial,
    )
    write_json(dataset_path / ".dataset", metadata)
    return DatasetBuildResult(dataset_path, manifest, metadata, reports)


def build_dataset_metadata(
    dataset_path: Path,
    train_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
    *,
    train_virial: bool,
    allow_partial: bool,
) -> dict[str, Any]:
    """Build the Phase 2 manifest shape for compatibility callers."""

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
    """Write the exact extxyz schema consumed by GPUMD NEP training."""

    rendered: list[str] = []
    count = 0
    for structure in structures:
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
        if include_virial:
            if virial is None:
                raise ValueError("missing_required_virial")
            virial_array = np.asarray(virial, dtype=float)
            if virial_array.shape != (3, 3) or not np.isfinite(virial_array).all():
                raise ValueError("Dataset virial is invalid")
        else:
            virial_array = None

        pbc_text = " ".join("T" if flag else "F" for flag in pbc)
        lattice_text = " ".join(f"{value:.10f}" for value in lattice.flat)
        header = (
            f"energy={energy:.10f} pbc=\"{pbc_text}\" "
            f"Lattice=\"{lattice_text}\" "
            "Properties=species:S:1:pos:R:3:force:R:3"
        )
        if virial_array is not None:
            virial_text = " ".join(f"{value:.10f}" for value in virial_array.flat)
            header += f' virial="{virial_text}"'
        rendered.append(f"{len(species)}\n{header}\n")
        rendered.extend(
            f"{symbol:3s} {position[0]:15.10f} {position[1]:15.10f} {position[2]:15.10f} "
            f"{force[0]:15.10f} {force[1]:15.10f} {force[2]:15.10f}\n"
            for symbol, position, force in zip(species, positions, forces)
        )
        count += 1

    from nepflow.io.atomic import atomic_write_text

    atomic_write_text(output_path, "".join(rendered))
    return count


__all__ = [
    "DatasetBuildReport",
    "DatasetBuildResult",
    "DatasetSplit",
    "build_dataset_metadata",
    "build_training_dataset",
    "iter_labeled_structures",
    "write_nep_dataset",
]
# End of canonical dataset module.
