"""Compatibility adapters for the canonical training dataset boundary.

The authoritative dataset builder consumes parsed, identity-bound VASP result
records.  This module is the only legacy adapter that resolves selected ASE
structures against the old filesystem job layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from ase.atoms import Atoms
from ase.io import read as ase_read

from nepflow.dft.vasp.inputs import (
    build_input_context,
    identity_for_structure,
    read_identity,
)
from nepflow.dft.vasp.outputs import (
    VaspJobEvidence,
    VaspRegistryEvidence,
    parse_outcar_result,
    resolve_verified_output,
)
from nepflow.dft.vasp.registry import (
    get_nepflow_root,
    get_registry_entry,
    read_completed_registry,
    read_status,
)
from nepflow.domain.identities import calculate_structure_id
from nepflow.domain.units import (
    ENERGY_UNIT_EV,
    FORCE_UNIT_EV_PER_ANGSTROM,
    VIRIAL_CONVENTION_POSITIVE_COMPRESSION,
    VIRIAL_UNIT_EV,
)
from nepflow.stages.training.dataset import DatasetBuildReport, DatasetSplit, write_nep_dataset
from nepflow.stages.training import dataset as canonical_dataset


def _reject(report: DatasetBuildReport, reason: str, *, member_index: int) -> None:
    report.record_rejection(reason, member_index=member_index)


def _extract_debug_labels(atoms: Atoms) -> dict[str, Any]:
    if "energy" in atoms.info:
        energy = float(atoms.info["energy"])
    elif atoms.calc is not None:
        energy = float(atoms.get_potential_energy())
    else:
        raise ValueError("Debug structure is missing an energy label")

    if "forces" in atoms.arrays:
        forces = np.asarray(atoms.arrays["forces"], dtype=float)
    elif "force" in atoms.arrays:
        forces = np.asarray(atoms.arrays["force"], dtype=float)
    elif atoms.calc is not None:
        forces = np.asarray(atoms.get_forces(), dtype=float)
    else:
        raise ValueError("Debug structure is missing force labels")

    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError("Debug structure force labels are invalid")
    if not np.isfinite(energy):
        raise ValueError("Debug structure energy label is not finite")
    return {
        "energy": energy,
        "forces": forces,
        "positions": atoms.get_positions(),
        "lattice": atoms.get_cell().array,
        "species": atoms.get_chemical_symbols(),
        "pbc": atoms.pbc.tolist(),
        "virial": atoms.info.get("virial"),
        "energy_unit": ENERGY_UNIT_EV,
        "force_unit": FORCE_UNIT_EV_PER_ANGSTROM,
        "virial_unit": VIRIAL_UNIT_EV,
        "virial_convention": VIRIAL_CONVENTION_POSITIVE_COMPRESSION,
    }


def resolve_legacy_labeled_structures(
    ase_structures: Sequence[Atoms],
    split: DatasetSplit | str,
    project_dir: Path,
    *,
    require_virial: bool = False,
    debug: bool = False,
    report: DatasetBuildReport | None = None,
    reader: Any = None,
) -> Iterator[dict[str, Any]]:
    """Resolve legacy selected structures for compatibility callers only."""

    selected_split = DatasetSplit.coerce(split)
    if report is None:
        report = DatasetBuildReport(selected_split)
        report.reset(len(ase_structures))
    elif report.split != selected_split:
        raise ValueError("Dataset report split does not match the input split")
    elif report.requested_count == 0 and not report.requested_members:
        report.reset(len(ase_structures))
    elif report.requested_count != len(ase_structures):
        raise ValueError("Dataset report requested_count does not match input")

    reader = reader or ase_read
    jobs_path = Path(project_dir) / "vasp" / "jobs" / selected_split.value
    input_context_record = build_input_context(Path(project_dir))

    if not jobs_path.exists():
        if not debug:
            for ordinal, atoms in enumerate(ase_structures):
                member_index = report.record_requested(
                    calculate_structure_id(atoms), ordinal
                )
                _reject(report, "vasp_jobs_path_missing", member_index=member_index)
            return
        for ordinal, atoms in enumerate(ase_structures):
            member_index = report.record_requested(
                calculate_structure_id(atoms), ordinal
            )
            synthetic = _extract_debug_labels(atoms)
            if require_virial and synthetic.get("virial") is None:
                raise ValueError("missing_required_virial")
            synthetic.update(
                {
                    "record_type": "debug_synthetic",
                    "split": selected_split.value,
                    "ordinal": ordinal,
                    "structure_id": calculate_structure_id(atoms),
                }
            )
            report.record_acceptance(content_record=synthetic, member_index=member_index)
            yield synthetic
        return

    if input_context_record is None:
        for ordinal, atoms in enumerate(ase_structures):
            member_index = report.record_requested(
                calculate_structure_id(atoms), ordinal
            )
            _reject(report, "vasp_input_context_missing", member_index=member_index)
        return

    input_context = input_context_record.as_legacy_mapping()
    input_context["registry"] = read_completed_registry(get_nepflow_root(project_dir))

    for ordinal, atoms in enumerate(ase_structures):
        member_index = report.record_requested(
            calculate_structure_id(atoms), ordinal
        )
        try:
            identity = identity_for_structure(atoms, input_context).as_dict()
            preferred = jobs_path / f"struct_{ordinal:04d}"
            candidate_dirs = [preferred]
            candidate_dirs.extend(
                directory
                for directory in sorted(
                    (
                        directory
                        for directory in jobs_path.iterdir()
                        if directory.is_dir() and directory.name.startswith("struct_")
                    ),
                    key=lambda directory: directory.name,
                )
                if directory != preferred
            )
            current_jobs = [
                VaspJobEvidence(
                    job_directory=directory,
                    identity=read_identity(directory) or None,
                    status=read_status(directory),
                )
                for directory in candidate_dirs
                if directory.exists()
            ]
            registry_entry = get_registry_entry(
                input_context["registry"],
                identity["incar_hash"],
                identity["potcar_hash"],
                identity["structure_id"],
            )
            registry_evidence = (
                VaspRegistryEvidence(
                    outcar_path=Path(registry_entry["job_path"]) / "OUTCAR",
                    calculation_identity=identity,
                )
                if registry_entry is not None
                else None
            )
            resolved_output = resolve_verified_output(
                identity,
                current_jobs=current_jobs,
                registry_evidence=registry_evidence,
            )
        except Exception as exc:
            _reject(
                report,
                f"identity_resolution_failed:{type(exc).__name__}",
                member_index=member_index,
            )
            continue

        if resolved_output is None:
            _reject(report, "completed_outcar_missing", member_index=member_index)
            continue

        try:
            result = parse_outcar_result(
                resolved_output.outcar_path,
                atoms,
                require_virial=require_virial,
                calculation_identity=identity,
                identity_evidence=resolved_output,
                reader=reader,
                identity_reader=read_identity,
            )
        except Exception as exc:
            _reject(
                report,
                f"extraction_failed:{type(exc).__name__}",
                member_index=member_index,
            )
            continue
        if not result.accepted:
            _reject(
                report,
                result.rejection_reason or "record_rejected",
                member_index=member_index,
            )
            continue
        if not result.source_outcar_hash:
            _reject(report, "missing_source_outcar_hash", member_index=member_index)
            continue
        report.record_acceptance(result, member_index=member_index)
        yield result.as_structure_dict()


def prepare_dataset(
    dataset_path: Path,
    ase_structures: Sequence[Atoms],
    is_train: bool,
    project_dir: Path,
    train_virial: bool = False,
    debug: bool = False,
    extraction_report: dict | None = None,
) -> int:
    """Deprecated single-split adapter for pre-Phase 4 callers."""

    split = DatasetSplit.TRAIN if is_train else DatasetSplit.TEST
    report = DatasetBuildReport(split, requested_count=len(ase_structures))
    structures = list(
        resolve_legacy_labeled_structures(
            ase_structures,
            split,
            project_dir,
            require_virial=train_virial,
            debug=debug,
            report=report,
            reader=canonical_dataset.ase_read or ase_read,
        )
    )
    count = write_nep_dataset(
        dataset_path,
        structures,
        include_virial=train_virial,
    )
    if extraction_report is not None:
        extraction_report.clear()
        extraction_report.update(report.to_dict())
    return count


__all__ = ["prepare_dataset", "resolve_legacy_labeled_structures"]
