"""Prepare sub-stage: parse OUTCAR files and write XYZ datasets."""

from pathlib import Path
from typing import Generator, List

from ase.atoms import Atoms
from ase.io import read as ase_read

import numpy as np

from nepflow.dft.vasp.inputs import (
    build_input_context,
    identity_for_structure,
    read_identity,
)
from nepflow.dft.vasp.outputs import (
    ResolvedVaspOutput,
    VaspParseResult,
    outcar_is_complete,
    parse_outcar_result as _canonical_parse_outcar_result,
    parse_virial_from_outcar,
)

from ..run_vasp._common import (
    get_nepflow_root,
    get_registry_entry,
    read_completed_registry,
    read_status,
)
from ._common import HAS_TQDM, logger, tqdm


def prepare_dataset(
    dataset_path: Path,
    ase_structures: List[Atoms],
    is_train: bool,
    project_dir: Path,
    train_virial: bool = False,
    debug: bool = False,
    extraction_report: dict | None = None,
) -> int:
    """Parse OUTCAR files and write XYZ dataset in streaming mode.
    
    Combines parsing and writing for memory efficiency with large datasets.
    
    Args:
        dataset_path: Output path for train.xyz or test.xyz
        ase_structures: List of ASE Atoms objects from selected structures
        is_train: Whether this is train (True) or test (False) dataset
        project_dir: Project root directory
        train_virial: Whether to include virial tensor
        debug: Whether to use synthetic data if OUTCAR missing
        
    Returns:
        Number of structures successfully written
    """
    dataset_type = "train" if is_train else "test"
    logger.info(f"Processing {len(ase_structures)} {dataset_type} structures")

    report = extraction_report if extraction_report is not None else {}
    report.clear()
    report.update(
        {
            "requested_count": len(ase_structures),
            "accepted_results": [],
            "accepted_content_records": [],
            "rejected_reason_counts": {},
        }
    )

    # Create generator for parsing (doesn't load all into memory)
    structure_generator = _parse_structures(
        ase_structures,
        is_train=is_train,
        project_dir=project_dir,
        require_virial=train_virial,
        debug=debug,
        extraction_report=report,
    )

    # Wrap generator with progress bar if tqdm available
    if HAS_TQDM:
        structure_generator = tqdm(
            structure_generator,
            total=len(ase_structures),
            desc=f"Writing {dataset_type}.xyz",
            unit=" structures",
        )

    # Write XYZ file, streaming structures one at a time
    count = _write_xyz_file(dataset_path, structure_generator, include_virial=train_virial)

    report["accepted_count"] = count
    report["rejected_count"] = report["requested_count"] - count

    logger.info(f"Wrote {count}/{len(ase_structures)} {dataset_type} structures to {dataset_path.name}")

    return count


def _parse_structures(
    ase_structures: List[Atoms],
    is_train: bool,
    project_dir: Path,
    require_virial: bool = False,
    debug: bool = False,
    extraction_report: dict | None = None,
) -> Generator[dict, None, None]:
    """Generator: parse structures from ASE Atoms objects and find corresponding OUTCAR files.
    
    Streams results one at a time to avoid memory buildup with large datasets.
    """
    dataset_type = "train" if is_train else "test"
    vasp_jobs_path = project_dir / "vasp" / "jobs" / dataset_type
    input_context = _build_vasp_input_context(project_dir)

    if not vasp_jobs_path.exists():
        logger.warning(f"VASP jobs path not found: {vasp_jobs_path}")
        if not debug:
            logger.info("Cannot find OUTCAR files. Skipping this dataset.")
            for _ in ase_structures:
                _record_rejection(extraction_report, "vasp_jobs_path_missing")
            return
        # In debug mode, yield synthetic data
        for struct_idx, atoms in enumerate(ase_structures):
            logger.debug(f"  [{dataset_type}] Generating synthetic structure {struct_idx}")
            synthetic = _extract_from_atoms(atoms)
            if extraction_report is not None:
                extraction_report.setdefault("accepted_content_records", []).append(
                    {
                        "record_type": "debug_synthetic",
                        "split": dataset_type,
                        "ordinal": struct_idx,
                        "energy": synthetic["energy"],
                        "forces": np.asarray(synthetic["forces"]).tolist(),
                        "positions": np.asarray(synthetic["positions"]).tolist(),
                        "lattice": np.asarray(synthetic["lattice"]).tolist(),
                        "species": list(synthetic["species"]),
                        "pbc": list(synthetic["pbc"]),
                    }
                )
            yield synthetic
        return

    if input_context is None:
        logger.warning("Cannot build VASP input hashes. Skipping this dataset.")
        for _ in ase_structures:
            _record_rejection(extraction_report, "vasp_input_context_missing")
        return

    for struct_idx, atoms in enumerate(ase_structures):
        try:
            resolved_output = _resolve_outcar_for_structure(
                atoms,
                dataset_type,
                struct_idx,
                project_dir,
                vasp_jobs_path,
                input_context,
            )
        except Exception as e:
            logger.warning(f"[{dataset_type}] Skipping struct_{struct_idx:04d}: {e}")
            _record_rejection(
                extraction_report,
                f"identity_resolution_failed:{type(e).__name__}",
            )
            continue

        if resolved_output is None:
            logger.warning(f"[{dataset_type}] Skipping struct_{struct_idx:04d}: no completed OUTCAR found")
            _record_rejection(extraction_report, "completed_outcar_missing")
            continue

        try:
            logger.debug(f"[{dataset_type}] Parsing struct_{struct_idx:04d}")
            identity = _identity_for_structure(atoms, input_context)
            result = _parse_outcar_result(
                resolved_output.outcar_path,
                atoms,
                require_virial=require_virial,
                calculation_identity=identity,
                identity_evidence=resolved_output,
            )
            if result.accepted:
                _record_acceptance(extraction_report, result)
                yield result.as_structure_dict()
            else:
                _record_rejection(extraction_report, result.rejection_reason or "record_rejected")
                logger.warning(
                    "[%s] Rejecting struct_%04d: %s",
                    dataset_type,
                    struct_idx,
                    result.rejection_reason,
                )
        except Exception as e:
            logger.warning(f"[{dataset_type}] Skipping struct_{struct_idx:04d}: {e}")
            _record_rejection(extraction_report, f"extraction_failed:{type(e).__name__}")


def _record_acceptance(report: dict | None, result: VaspParseResult) -> None:
    """Record an accepted immutable parse result for manifest finalization."""
    if report is not None:
        report.setdefault("accepted_results", []).append(result)


def _record_rejection(report: dict | None, reason: str) -> None:
    """Record one machine-readable extraction rejection."""
    if report is not None:
        reasons = report.setdefault("rejected_reason_counts", {})
        reasons[reason] = reasons.get(reason, 0) + 1


def _build_vasp_input_context(project_dir: Path) -> dict | None:
    """Compatibility bridge to the canonical project input context."""
    context = build_input_context(project_dir)
    if context is None:
        return None
    result = context.as_legacy_mapping()
    result["registry"] = read_completed_registry(get_nepflow_root(project_dir))
    return result


def _resolve_outcar_for_structure(
    atoms: Atoms,
    dataset_type: str,
    struct_idx: int,
    project_dir: Path,
    vasp_jobs_path: Path,
    input_context: dict,
) -> ResolvedVaspOutput | None:
    """Resolve the completed OUTCAR for one selected structure by input hashes."""
    identity = _identity_for_structure(atoms, input_context)
    structure_id = identity["structure_id"]
    incar_hash = identity["incar_hash"]
    potcar_hash = identity["potcar_hash"]

    preferred = vasp_jobs_path / f"struct_{struct_idx:04d}"
    resolved_output = _outcar_from_local_job(preferred, identity)
    if resolved_output is not None:
        return resolved_output

    for struct_dir in sorted(
        [d for d in vasp_jobs_path.iterdir() if d.is_dir() and d.name.startswith("struct_")],
        key=lambda d: d.name,
    ):
        if struct_dir == preferred:
            continue
        resolved_output = _outcar_from_local_job(struct_dir, identity)
        if resolved_output is not None:
            return resolved_output

    entry = get_registry_entry(
        input_context["registry"],
        incar_hash,
        potcar_hash,
        structure_id,
    )
    if entry and entry.get("job_path"):
        outcar = Path(entry["job_path"]) / "OUTCAR"
        if outcar_is_complete(outcar):
            return ResolvedVaspOutput(
                outcar_path=outcar,
                calculation_identity=tuple(sorted(identity.items())),
                verification_source="completed_registry_key",
            )

    return None


def _identity_for_structure(atoms: Atoms, input_context: dict) -> dict[str, str]:
    """Compatibility bridge to the typed canonical VASP identity."""
    return identity_for_structure(atoms, input_context).as_dict()


def _outcar_from_local_job(
    struct_dir: Path,
    identity: dict,
) -> ResolvedVaspOutput | None:
    """Return a completed OUTCAR from a local identity-matched job folder."""
    if not struct_dir.exists():
        return None
    local_identity = _read_identity(struct_dir)
    if not local_identity or any(local_identity.get(k) != v for k, v in identity.items()):
        return None

    status = read_status(struct_dir)
    if status.get("status") == "reused" and status.get("reused_from"):
        reused_outcar = Path(status["reused_from"]) / "OUTCAR"
        if outcar_is_complete(reused_outcar):
            return ResolvedVaspOutput(
                outcar_path=reused_outcar,
                calculation_identity=tuple(sorted(identity.items())),
                verification_source="current_job_identity_reuse",
            )

    outcar = struct_dir / "OUTCAR"
    if not outcar_is_complete(outcar):
        return None
    return ResolvedVaspOutput(
        outcar_path=outcar,
        calculation_identity=tuple(sorted(identity.items())),
        verification_source="current_job_identity",
    )


def _read_identity(struct_dir: Path) -> dict:
    """Compatibility bridge for legacy callers of the canonical reader."""
    return read_identity(struct_dir)


def _parse_outcar(
    outcar_path: Path,
    ase_atoms: Atoms,
    require_virial: bool = False,
    calculation_identity: dict[str, str] | None = None,
) -> dict | None:
    """Return legacy structure data only when the authoritative parse succeeds."""
    result = _parse_outcar_result(
        outcar_path,
        ase_atoms,
        require_virial=require_virial,
        calculation_identity=calculation_identity,
    )
    if not result.accepted:
        logger.warning("Rejected %s: %s", outcar_path, result.rejection_reason)
        return None
    return result.as_structure_dict()


def _parse_outcar_result(
    outcar_path: Path,
    ase_atoms: Atoms,
    require_virial: bool = False,
    calculation_identity: dict[str, str] | None = None,
    identity_evidence: ResolvedVaspOutput | None = None,
) -> VaspParseResult:
    """Compatibility bridge to the canonical VASP output parser."""
    return _canonical_parse_outcar_result(
        outcar_path,
        ase_atoms,
        require_virial=require_virial,
        calculation_identity=calculation_identity,
        identity_evidence=identity_evidence,
        reader=ase_read,
        identity_reader=_read_identity,
    )


def _extract_from_atoms(atoms: Atoms) -> dict:
    """Extract already-materialized labels without fabricating placeholders."""
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
        "virial": atoms.info.get("virial", None),
    }


def _write_xyz_file(
    output_path: Path,
    structures: Generator[dict, None, None],
    include_virial: bool = False,
) -> int:
    """Write extended XYZ file with NEP format, processing structures from generator.
    
    Streams structures one at a time to avoid memory buildup.
    Output format matches GPUMD NEP training requirements:
    - Line 1: number of atoms
    - Line 2: energy, pbc, Lattice, Properties (and optional virial)
    - Lines 3+: species, position (x, y, z), force (fx, fy, fz)
    
    Returns:
        Number of structures written
    """
    count = 0
    with open(output_path, "w") as f:
        for structure in structures:
            n_atoms = len(structure["species"])
            f.write(f"{n_atoms}\n")

            # Build header in correct order for NEP
            energy = structure["energy"]
            pbc = structure["pbc"]
            lattice = structure["lattice"]

            # Format PBC vector: "T T T" or "T T F" etc
            pbc_str = " ".join("T" if p else "F" for p in pbc)

            # Format Lattice as flattened 3x3: "ax ay az bx by bz cx cy cz"
            lattice_str = " ".join(f"{v:.10f}" for v in lattice.flat)

            # Properties descriptor: species, position, force
            properties_str = "species:S:1:pos:R:3:force:R:3"

            # Build header line in NEP format
            header = (
                f"energy={energy:.10f} "
                f'pbc="{pbc_str}" '
                f'Lattice="{lattice_str}" '
                f"Properties={properties_str}"
            )

            # Add virial if requested and available
            if include_virial and structure["virial"] is not None:
                virial_str = " ".join(f"{v:.10f}" for v in structure["virial"].flat)
                header += f' virial="{virial_str}"'

            f.write(header + "\n")

            # Write atoms: species followed by position (3 floats) then force (3 floats)
            for species, pos, force in zip(
                structure["species"],
                structure["positions"],
                structure["forces"],
            ):
                f.write(
                    f"{species:3s} {pos[0]:15.10f} {pos[1]:15.10f} {pos[2]:15.10f} "
                    f"{force[0]:15.10f} {force[1]:15.10f} {force[2]:15.10f}\n"
                )

            count += 1

    return count
