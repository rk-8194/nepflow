"""Prepare sub-stage: parse OUTCAR files and write XYZ datasets."""

import json
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List

from ase.atoms import Atoms
from ase.io import read as ase_read

import numpy as np

from ..run_vasp._common import (
    get_nepflow_root,
    get_registry_entry,
    file_sha256,
    hash_incar_text,
    hash_potcar_bytes,
    hash_structure,
    outcar_is_complete,
    parse_virial_from_outcar,
    read_completed_registry,
    read_status,
)
from ..run_vasp.prepare import inject_incar_defaults

from ._common import HAS_TQDM, StructureValidationError, logger, tqdm, validate_structure


@dataclass(frozen=True)
class VaspParseResult:
    """Immutable, authoritative result of parsing one VASP OUTCAR."""

    structure_id: str
    calculation_identity: tuple[tuple[str, str], ...]
    source_outcar: str
    source_outcar_hash: str | None
    status: str
    rejection_reason: str | None
    energy_ev: float | None
    forces_ev_per_angstrom: np.ndarray | None
    virial_ev: np.ndarray | None
    positions_angstrom: np.ndarray | None
    lattice_angstrom: np.ndarray | None
    species: tuple[str, ...]
    pbc: tuple[bool, ...]
    energy_unit: str = "eV"
    force_unit: str = "eV/Angstrom"
    virial_unit: str = "eV"
    virial_convention: str = "positive_compression"

    def __post_init__(self) -> None:
        for field_name in (
            "forces_ev_per_angstrom",
            "virial_ev",
            "positions_angstrom",
            "lattice_angstrom",
        ):
            value = getattr(self, field_name)
            if value is not None:
                immutable = np.array(value, dtype=float, copy=True)
                immutable.setflags(write=False)
                object.__setattr__(self, field_name, immutable)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def as_structure_dict(self) -> dict:
        if not self.accepted:
            raise ValueError(self.rejection_reason or "rejected_parse_result")
        return {
            "energy": self.energy_ev,
            "forces": self.forces_ev_per_angstrom,
            "positions": self.positions_angstrom,
            "lattice": self.lattice_angstrom,
            "species": list(self.species),
            "pbc": list(self.pbc),
            "virial": self.virial_ev,
            "structure_id": self.structure_id,
            "calculation_identity": dict(self.calculation_identity),
            "source_outcar": self.source_outcar,
            "source_outcar_hash": self.source_outcar_hash,
            "energy_unit": self.energy_unit,
            "force_unit": self.force_unit,
            "virial_unit": self.virial_unit,
            "virial_convention": self.virial_convention,
        }


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
            yield _extract_from_atoms(atoms)
        return

    if input_context is None:
        logger.warning("Cannot build VASP input hashes. Skipping this dataset.")
        for _ in ase_structures:
            _record_rejection(extraction_report, "vasp_input_context_missing")
        return

    for struct_idx, atoms in enumerate(ase_structures):
        try:
            outcar_path = _resolve_outcar_for_structure(
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

        if outcar_path is None:
            logger.warning(f"[{dataset_type}] Skipping struct_{struct_idx:04d}: no completed OUTCAR found")
            _record_rejection(extraction_report, "completed_outcar_missing")
            continue

        try:
            logger.debug(f"[{dataset_type}] Parsing struct_{struct_idx:04d}")
            identity = _identity_for_structure(atoms, input_context)
            result = _parse_outcar_result(
                outcar_path,
                atoms,
                require_virial=require_virial,
                calculation_identity=identity,
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
    """Build reusable hashes for the current project's VASP input files."""
    vasp_config_dir = project_dir / "config" / "vasp"
    incar_template = vasp_config_dir / "INCAR"
    if not incar_template.exists():
        return None

    config = ConfigParser()
    for candidate in (
        project_dir / "config" / "project.config",
        project_dir / "config" / f"{project_dir.name.removeprefix('project_')}.ini",
    ):
        if candidate.exists():
            config.read(candidate)
            break

    incar_text = inject_incar_defaults(
        incar_template.read_text(encoding="utf-8"),
        config,
    )
    potcar_data = {
        potcar_path.name.split("_", 1)[1]: potcar_path.read_bytes()
        for potcar_path in vasp_config_dir.glob("POTCAR_*")
    }
    return {
        "incar_hash": hash_incar_text(incar_text),
        "potcar_data": potcar_data,
        "registry": read_completed_registry(get_nepflow_root(project_dir)),
    }


def _resolve_outcar_for_structure(
    atoms: Atoms,
    dataset_type: str,
    struct_idx: int,
    project_dir: Path,
    vasp_jobs_path: Path,
    input_context: dict,
) -> Path | None:
    """Resolve the completed OUTCAR for one selected structure by input hashes."""
    identity = _identity_for_structure(atoms, input_context)
    structure_hash = identity["structure_hash"]
    incar_hash = identity["incar_hash"]
    potcar_hash = identity["potcar_hash"]

    preferred = vasp_jobs_path / f"struct_{struct_idx:04d}"
    outcar = _outcar_from_local_job(preferred, identity)
    if outcar is not None:
        return outcar

    for struct_dir in sorted(
        [d for d in vasp_jobs_path.iterdir() if d.is_dir() and d.name.startswith("struct_")],
        key=lambda d: d.name,
    ):
        if struct_dir == preferred:
            continue
        outcar = _outcar_from_local_job(struct_dir, identity)
        if outcar is not None:
            return outcar

    entry = get_registry_entry(
        input_context["registry"],
        incar_hash,
        potcar_hash,
        structure_hash,
    )
    if entry and entry.get("job_path"):
        outcar = Path(entry["job_path"]) / "OUTCAR"
        if outcar_is_complete(outcar):
            return outcar

    return None


def _identity_for_structure(atoms: Atoms, input_context: dict) -> dict[str, str]:
    """Build the exact VASP calculation identity for a selected structure."""
    structure_hash = hash_structure(atoms)
    potcar_data = input_context["potcar_data"]
    struct_elements = sorted(set(atoms.get_chemical_symbols()))
    missing = [elem for elem in struct_elements if elem not in potcar_data]
    if missing:
        raise FileNotFoundError(f"missing POTCAR files for: {', '.join(missing)}")
    potcar_hash = hash_potcar_bytes(
        b"".join(potcar_data[elem] for elem in struct_elements)
    )
    return {
        "structure_hash": structure_hash,
        "incar_hash": input_context["incar_hash"],
        "potcar_hash": potcar_hash,
    }


def _outcar_from_local_job(struct_dir: Path, identity: dict) -> Path | None:
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
            return reused_outcar

    outcar = struct_dir / "OUTCAR"
    return outcar if outcar_is_complete(outcar) else None


def _read_identity(struct_dir: Path) -> dict:
    identity_path = struct_dir / ".vasp_identity"
    if not identity_path.exists():
        return {}
    try:
        data = json.loads(identity_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


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
) -> VaspParseResult:
    """Parse one OUTCAR into an immutable accepted/rejected result."""
    structure_id = hash_structure(ase_atoms)
    identity = calculation_identity or {"structure_hash": structure_id}
    identity_items = tuple(sorted((str(key), str(value)) for key, value in identity.items()))
    source_outcar = str(outcar_path.resolve())
    source_hash = file_sha256(outcar_path)

    def rejected(reason: str) -> VaspParseResult:
        return VaspParseResult(
            structure_id=structure_id,
            calculation_identity=identity_items,
            source_outcar=source_outcar,
            source_outcar_hash=source_hash,
            status="rejected",
            rejection_reason=reason,
            energy_ev=None,
            forces_ev_per_angstrom=None,
            virial_ev=None,
            positions_angstrom=None,
            lattice_angstrom=None,
            species=(),
            pbc=(),
        )

    if calculation_identity is not None:
        source_identity = _read_identity(outcar_path.parent)
        if source_identity and any(
            source_identity.get(key) != value for key, value in calculation_identity.items()
        ):
            return rejected("incompatible_calculation_identity")

    try:
        atoms = ase_read(str(outcar_path))
    except Exception as exc:
        return rejected(f"outcar_parse_failed:{type(exc).__name__}:{exc}")

    expected_species = tuple(ase_atoms.get_chemical_symbols())
    actual_species = tuple(atoms.get_chemical_symbols())
    if len(atoms) != len(ase_atoms):
        return rejected(f"atom_count_mismatch:expected={len(ase_atoms)}:actual={len(atoms)}")
    if actual_species != expected_species:
        return rejected(f"species_mismatch:expected={expected_species}:actual={actual_species}")

    try:
        energy = float(atoms.get_potential_energy())
    except Exception as exc:
        return rejected(f"missing_energy:{type(exc).__name__}")
    try:
        forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception as exc:
        return rejected(f"missing_forces:{type(exc).__name__}")

    try:
        positions = np.asarray(atoms.get_positions(), dtype=float)
        lattice = np.asarray(atoms.get_cell().array, dtype=float)
        pbc = atoms.pbc.tolist()
        volume = float(atoms.get_volume())
    except Exception as exc:
        return rejected(f"invalid_structure_geometry:{type(exc).__name__}")

    virial = parse_virial_from_outcar(outcar_path, volume)
    if require_virial and virial is None:
        return rejected("missing_required_virial")

    structure = {
        "energy": energy,
        "forces": forces,
        "positions": positions,
        "lattice": lattice,
        "species": list(actual_species),
        "pbc": pbc,
        "virial": virial,
    }
    try:
        validate_structure(structure)
    except StructureValidationError as exc:
        return rejected(f"invalid_dft_labels:{exc}")

    return VaspParseResult(
        structure_id=structure_id,
        calculation_identity=identity_items,
        source_outcar=source_outcar,
        source_outcar_hash=source_hash,
        status="accepted",
        rejection_reason=None,
        energy_ev=energy,
        forces_ev_per_angstrom=forces,
        virial_ev=virial,
        positions_angstrom=positions,
        lattice_angstrom=lattice,
        species=actual_species,
        pbc=tuple(bool(value) for value in pbc),
    )


def _extract_from_atoms(atoms: Atoms) -> dict:
    """Extract structure data from ASE Atoms object."""
    result = {
        "energy": atoms.get_potential_energy() if "energy" in atoms.info or hasattr(atoms, "calc") else -1.0,
        "forces": atoms.get_forces() if "forces" in atoms.arrays else np.zeros((len(atoms), 3)),
        "positions": atoms.get_positions(),
        "lattice": atoms.get_cell().array,
        "species": atoms.get_chemical_symbols(),
        "pbc": atoms.pbc.tolist(),
        "virial": atoms.info.get("virial", None),
    }
    return result


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
