"""Canonical VASP output completion, parsing, and performance evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Callable, Iterable, Mapping, cast

import numpy as np
from ase.atoms import Atoms
from ase.io import read as ase_read

from nepflow.domain.identities import (
    calculate_structure_id,
    normalise_dft_calculation_identity,
)
from nepflow.domain.units import (
    ENERGY_UNIT_EV,
    FORCE_UNIT_EV_PER_ANGSTROM,
    VIRIAL_CONVENTION_POSITIVE_COMPRESSION,
    VIRIAL_UNIT_EV,
    stress_kbar_to_ev_per_angstrom3,
    virial_from_stress,
)
from nepflow.errors import StateError
from nepflow.io.hashing import sha256_file

from .inputs import read_identity

VASP_COMPLETION_MARKERS = ("General timing", "Voluntary context switches")


def is_completed_text(outcar_text: str) -> bool:
    """Check completion markers in already-read OUTCAR text."""
    return any(marker in outcar_text[-2000:] for marker in VASP_COMPLETION_MARKERS)


def outcar_is_complete(outcar_path: Path) -> bool:
    """Check whether an OUTCAR tail contains a VASP completion marker."""
    outcar_path = Path(outcar_path)
    if not outcar_path.exists():
        return False
    try:
        with open(outcar_path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 50_000))
            tail = handle.read()
        return any(marker in tail for marker in VASP_COMPLETION_MARKERS)
    except OSError:
        return False


_STRESS_PATTERN = re.compile(
    r"STRESS\s+in cartesian coordinates \(kB\)\n"
    r"\s+([-+\.\dEe]+)\s+([-+\.\dEe]+)\s+([-+\.\dEe]+)\n"
    r"\s+([-+\.\dEe]+)\s+([-+\.\dEe]+)\s+([-+\.\dEe]+)\n"
    r"\s+([-+\.\dEe]+)\s+([-+\.\dEe]+)\s+([-+\.\dEe]+)"
)
_STRESS_FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")


def parse_stress_from_outcar(
    outcar_path: Path,
    *,
    prefer_ase: bool = True,
) -> np.ndarray | None:
    """Parse the final VASP stress tensor in eV/Angstrom^3.

    ASE handles common OUTCAR variants.  The explicit text paths retain the
    verified Phase 2 handling for the kB matrix layouts used by existing
    utilities and fixtures.  The returned tensor keeps the backend-native
    stress sign; conversion to NEPFlow's positive-compression virial remains
    the responsibility of :func:`parse_virial_from_outcar`.
    """
    outcar_path = Path(outcar_path)
    if prefer_ase:
        try:
            atoms = cast(Atoms, ase_read(str(outcar_path), format="vasp-out"))
            stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
            if stress.shape == (3, 3) and np.all(np.isfinite(stress)):
                return stress
        except Exception:
            # Continue to the established text parser for unusual layouts
            # that ASE does not accept.
            pass

    try:
        outcar_text = outcar_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    return _parse_text_stress(outcar_text)


def _parse_text_stress(outcar_text: str) -> np.ndarray | None:
    """Parse the accepted non-ASE stress layouts without changing units."""

    matches = list(_STRESS_PATTERN.finditer(outcar_text))
    if matches:
        values = [float(matches[-1].group(index)) for index in range(1, 10)]
        return stress_kbar_to_ev_per_angstrom3(np.asarray(values, dtype=float).reshape(3, 3))

    # Preserve the two alternate text layouts accepted by the elastic
    # analysis utility without making that utility another OUTCAR parser.
    last_matrix: np.ndarray | None = None
    lines = outcar_text.splitlines()
    for index, line in enumerate(lines):
        if "STRESS" not in line.upper() or "KB" not in line.upper():
            continue
        rows: list[list[float]] = []
        for probe in lines[index + 1 : index + 8]:
            numbers = [float(value) for value in _STRESS_FLOAT_PATTERN.findall(probe)]
            if len(numbers) >= 3:
                rows.append(numbers[:3])
            if len(rows) == 3:
                break
        if len(rows) == 3:
            last_matrix = np.asarray(rows, dtype=float)
    if last_matrix is not None:
        return stress_kbar_to_ev_per_angstrom3(last_matrix)

    for line in reversed(lines):
        upper = line.upper()
        if "KB" not in upper or "STRESS" not in upper:
            continue
        numbers = [float(value) for value in _STRESS_FLOAT_PATTERN.findall(line)]
        if len(numbers) >= 6:
            xx, yy, zz, xy, yz, zx = numbers[-6:]
            return stress_kbar_to_ev_per_angstrom3(
                np.asarray(
                    [[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]],
                    dtype=float,
                )
            )
    return None


def parse_virial_from_outcar(outcar_path: Path, volume: float) -> np.ndarray | None:
    """Parse VASP stress and convert it to NEPFlow's virial convention."""
    try:
        # Keep the verified Phase 2 text semantics for the canonical virial
        # result even when ASE offers a different OUTCAR interpretation.
        stress = parse_stress_from_outcar(outcar_path, prefer_ase=False)
        if stress is None:
            return None
        return virial_from_stress(stress, float(volume))
    except (TypeError, ValueError):
        return None


class DftOutputValidationError(ValueError):
    """Raised when VASP output lacks complete scientific labels."""


def validate_dft_result_labels(structure: Mapping[str, object]) -> bool:
    """Validate VASP labels with the accepted Phase 2 error vocabulary."""
    energy = structure.get("energy")
    if not isinstance(energy, Real) or not np.isfinite(float(energy)):
        raise DftOutputValidationError("missing_or_nonfinite_energy")

    species = structure.get("species")
    if not isinstance(species, (list, tuple)) or not species:
        raise DftOutputValidationError("missing_species")

    try:
        forces = np.asarray(structure["forces"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise DftOutputValidationError("missing_or_invalid_forces") from exc
    if forces.shape != (len(species), 3):
        raise DftOutputValidationError("forces_shape_mismatch")
    if not np.all(np.isfinite(forces)):
        raise DftOutputValidationError("nonfinite_forces")

    try:
        lattice = np.asarray(structure["lattice"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise DftOutputValidationError("missing_or_invalid_lattice") from exc
    if lattice.shape != (3, 3):
        raise DftOutputValidationError("lattice_shape_mismatch")
    if not np.all(np.isfinite(lattice)):
        raise DftOutputValidationError("nonfinite_lattice")

    virial = structure.get("virial")
    if virial is not None:
        try:
            virial_array = np.asarray(virial, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DftOutputValidationError("invalid_virial") from exc
        if virial_array.shape != (3, 3):
            raise DftOutputValidationError("virial_shape_mismatch")
        if not np.all(np.isfinite(virial_array)):
            raise DftOutputValidationError("nonfinite_virial")
    return True


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
    energy_unit: str = ENERGY_UNIT_EV
    force_unit: str = FORCE_UNIT_EV_PER_ANGSTROM
    virial_unit: str = VIRIAL_UNIT_EV
    virial_convention: str = VIRIAL_CONVENTION_POSITIVE_COMPRESSION

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


@dataclass(frozen=True)
class ResolvedVaspOutput:
    """An OUTCAR whose calculation identity was verified during resolution."""

    outcar_path: Path
    calculation_identity: tuple[tuple[str, str], ...]
    verification_source: str


@dataclass(frozen=True, slots=True)
class VaspJobEvidence:
    """Identity/status evidence supplied by a current job-directory caller."""

    job_directory: Path
    identity: Mapping[str, object] | None
    status: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VaspRegistryEvidence:
    """An exact registry-key match supplied by the registry caller."""

    outcar_path: Path
    calculation_identity: Mapping[str, object]


_IDENTITY_KEYS = ("structure_id", "incar_hash", "potcar_hash", "calculation_id")


def _validated_identity(
    value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise StateError(f"{label} must be an object")
    normalized = normalise_dft_calculation_identity(value)
    for key in _IDENTITY_KEYS:
        item = normalized.get(key)
        if not isinstance(item, str) or not item.strip():
            raise StateError(f"{label} is missing a valid {key}")
    return {key: str(normalized[key]) for key in _IDENTITY_KEYS}


def _identity_items(identity: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(identity.items()))


def resolve_verified_output(
    expected_identity: Mapping[str, object],
    *,
    current_jobs: Iterable[VaspJobEvidence] = (),
    registry_evidence: VaspRegistryEvidence | None = None,
) -> ResolvedVaspOutput | None:
    """Resolve only outputs whose scientific identity is already evidenced.

    The caller owns directory enumeration, status-file reads, and registry
    persistence.  This function owns the trust rule: a current job must have
    a matching identity sidecar, while a reused or registry OUTCAR may be in a
    historical directory without a sidecar of its own.
    """
    expected = _validated_identity(expected_identity, label="expected VASP identity")
    expected_items = _identity_items(expected)

    for job in current_jobs:
        resolved = _resolve_current_job_output(job, expected, expected_items)
        if resolved is not None:
            return resolved

    if registry_evidence is not None:
        observed = _validated_identity(
            registry_evidence.calculation_identity,
            label="registry VASP identity",
        )
        if observed != expected:
            return None
        if outcar_is_complete(registry_evidence.outcar_path):
            return ResolvedVaspOutput(
                outcar_path=Path(registry_evidence.outcar_path),
                calculation_identity=expected_items,
                verification_source="completed_registry_key",
            )
    return None


def _resolve_current_job_output(
    job: VaspJobEvidence,
    expected: Mapping[str, str],
    expected_items: tuple[tuple[str, str], ...],
) -> ResolvedVaspOutput | None:
    """Resolve one current job only when its identity and completion agree."""

    if not isinstance(job.status, Mapping):
        raise StateError("VASP job status evidence must be an object")
    if job.identity is None:
        return None
    observed = _validated_identity(job.identity, label="current VASP identity")
    if observed != expected:
        return None

    status = job.status.get("status")
    reused_from = job.status.get("reused_from")
    if reused_from is not None and not isinstance(reused_from, str):
        raise StateError("VASP reused_from status evidence must be a path")
    if status == "reused" and reused_from:
        reused_outcar = Path(reused_from) / "OUTCAR"
        if outcar_is_complete(reused_outcar):
            return ResolvedVaspOutput(
                outcar_path=reused_outcar,
                calculation_identity=expected_items,
                verification_source="current_job_identity_reuse",
            )

    outcar = Path(job.job_directory) / "OUTCAR"
    if outcar_is_complete(outcar):
        return ResolvedVaspOutput(
            outcar_path=outcar,
            calculation_identity=expected_items,
            verification_source="current_job_identity",
        )
    return None


def parse_outcar_result(
    outcar_path: Path,
    ase_atoms: Atoms,
    require_virial: bool = False,
    calculation_identity: Mapping[str, str] | None = None,
    identity_evidence: ResolvedVaspOutput | None = None,
    *,
    reader: Callable[[str], Atoms] | None = None,
    identity_reader: Callable[[Path], dict] | None = None,
) -> VaspParseResult:
    """Parse one OUTCAR into an immutable accepted/rejected result."""
    if reader is None:
        reader = cast(Callable[[str], Atoms], ase_read)
    identity_reader = read_identity if identity_reader is None else identity_reader
    structure_id = calculate_structure_id(ase_atoms)
    identity_items_source = dict(calculation_identity or {})
    identity_items = tuple(
        sorted((str(key), str(value)) for key, value in identity_items_source.items())
    )
    source_outcar = str(Path(outcar_path).resolve())
    source_hash = sha256_file(Path(outcar_path), required=False)

    if calculation_identity is None:
        return _rejected_parse_result(
            structure_id,
            identity_items,
            source_outcar,
            source_hash,
            "missing_calculation_identity",
        )
    comparison_identity = _validated_identity(
        calculation_identity,
        label="expected VASP identity",
    )

    if identity_evidence is None:
        source_identity = identity_reader(Path(outcar_path).parent)
        if not source_identity:
            return _rejected_parse_result(
                structure_id,
                identity_items,
                source_outcar,
                source_hash,
                "missing_calculation_identity",
            )
        if any(source_identity.get(key) != value for key, value in comparison_identity.items()):
            return _rejected_parse_result(
                structure_id,
                identity_items,
                source_outcar,
                source_hash,
                "incompatible_calculation_identity",
            )
    else:
        if identity_evidence.outcar_path.resolve() != Path(outcar_path).resolve():
            return _rejected_parse_result(
                structure_id,
                identity_items,
                source_outcar,
                source_hash,
                "incompatible_calculation_identity",
            )
        try:
            evidence_identity = _validated_identity(
                dict(identity_evidence.calculation_identity),
                label="verified VASP identity",
            )
        except StateError:
            return _rejected_parse_result(
                structure_id,
                identity_items,
                source_outcar,
                source_hash,
                "incompatible_calculation_identity",
            )
        if evidence_identity != comparison_identity:
            return _rejected_parse_result(
                structure_id,
                identity_items,
                source_outcar,
                source_hash,
                "incompatible_calculation_identity",
            )

    try:
        atoms = reader(str(outcar_path))
    except Exception as exc:
        return _rejected_parse_result(
            structure_id,
            identity_items,
            source_outcar,
            source_hash,
            f"outcar_parse_failed:{type(exc).__name__}:{exc}",
        )

    parsed = _parse_vasp_structure_fields(
        atoms,
        ase_atoms,
        outcar_path,
        require_virial=require_virial,
    )
    if isinstance(parsed, str):
        return _rejected_parse_result(
            structure_id,
            identity_items,
            source_outcar,
            source_hash,
            parsed,
        )
    energy, forces, virial, positions, lattice, actual_species, pbc = parsed

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


def _rejected_parse_result(
    structure_id: str,
    identity_items: tuple[tuple[str, str], ...],
    source_outcar: str,
    source_hash: str | None,
    reason: str,
) -> VaspParseResult:
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


def _parse_vasp_structure_fields(
    atoms: Atoms,
    expected_atoms: Atoms,
    outcar_path: Path,
    *,
    require_virial: bool,
) -> (
    tuple[
        float,
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
        tuple[str, ...],
        list[bool],
    ]
    | str
):
    expected_species = tuple(expected_atoms.get_chemical_symbols())
    actual_species = tuple(atoms.get_chemical_symbols())
    if len(atoms) != len(expected_atoms):
        return f"atom_count_mismatch:expected={len(expected_atoms)}:actual={len(atoms)}"
    if actual_species != expected_species:
        return f"species_mismatch:expected={expected_species}:actual={actual_species}"

    try:
        energy = float(atoms.get_potential_energy())
    except Exception as exc:
        return f"missing_energy:{type(exc).__name__}"
    try:
        forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception as exc:
        return f"missing_forces:{type(exc).__name__}"

    try:
        positions = np.asarray(atoms.get_positions(), dtype=float)
        lattice = np.asarray(atoms.get_cell().array, dtype=float)
        pbc = atoms.pbc.tolist()
        volume = float(atoms.get_volume())
    except Exception as exc:
        return f"invalid_structure_geometry:{type(exc).__name__}"

    virial = parse_virial_from_outcar(outcar_path, volume)
    if require_virial and virial is None:
        return "missing_required_virial"

    try:
        validate_dft_result_labels(
            {
                "energy": energy,
                "forces": forces,
                "positions": positions,
                "lattice": lattice,
                "species": list(actual_species),
                "pbc": pbc,
                "virial": virial,
            }
        )
    except DftOutputValidationError as exc:
        return f"invalid_dft_labels:{exc}"
    return energy, forces, virial, positions, lattice, actual_species, pbc


def parse_outcar(
    outcar_path: Path,
    ase_atoms: Atoms,
    require_virial: bool = False,
    calculation_identity: Mapping[str, str] | None = None,
    identity_evidence: ResolvedVaspOutput | None = None,
    *,
    reader: Callable[[str], Atoms] | None = None,
    identity_reader: Callable[[Path], dict] | None = None,
) -> dict | None:
    """Return legacy structure data only for an accepted parse."""
    result = parse_outcar_result(
        outcar_path,
        ase_atoms,
        require_virial=require_virial,
        calculation_identity=calculation_identity,
        identity_evidence=identity_evidence,
        reader=reader,
        identity_reader=identity_reader,
    )
    return result.as_structure_dict() if result.accepted else None


@dataclass(frozen=True, slots=True)
class VaspPerformanceEvidence:
    """Performance fields parsed directly from one completed OUTCAR."""

    total_ranks: int
    loop_times: tuple[float, ...]
    irreducible_kpoints: int
    electrons: float
    mpi_ranks: int = 0

    @property
    def average_loop_time(self) -> float:
        return sum(self.loop_times) / len(self.loop_times) if self.loop_times else 0.0


def parse_performance_evidence(outcar_text: str) -> VaspPerformanceEvidence:
    """Parse canonical performance evidence for DFT reports and benchmarks."""
    ranks_match = re.search(r"running on\s+(\d+)\s+total cores", outcar_text)
    mpi_ranks_match = re.search(r"running\s+(\d+)\s+mpi-ranks", outcar_text)
    loops = tuple(
        float(match.group(1))
        for match in re.finditer(
            r"LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)",
            outcar_text,
        )
    )
    kpoints_match = re.search(r"Found\s+(\d+)\s+irreducible k-points", outcar_text)
    electrons_match = re.search(r"NELECT\s*=\s*([\d.]+)", outcar_text)
    return VaspPerformanceEvidence(
        total_ranks=int(ranks_match.group(1)) if ranks_match else 0,
        mpi_ranks=int(mpi_ranks_match.group(1)) if mpi_ranks_match else 0,
        loop_times=loops,
        irreducible_kpoints=int(kpoints_match.group(1)) if kpoints_match else 0,
        electrons=float(electrons_match.group(1)) if electrons_match else 0.0,
    )


class VaspMemoryParseError(ValueError):
    """Raised when a completed VASP memory record is malformed or unreadable."""


def parse_memory_record(outcar_path: Path, gpus_per_node: int) -> dict[str, object] | None:
    """Parse one eligible historical ``.vasp_memory`` row.

    Missing artifacts or an unfinished OUTCAR describe an absent historical
    record and return ``None``.  Once the OUTCAR is complete and the required
    inputs are present, malformed or unreadable evidence raises
    :class:`VaspMemoryParseError` so migration diagnostics cannot silently
    discard a completed run.
    """

    if gpus_per_node < 1:
        raise ValueError("gpus_per_node must be positive")
    outcar_path = Path(outcar_path)
    struct_dir = outcar_path.parent
    poscar = struct_dir / "POSCAR"
    incar = struct_dir / "INCAR"
    if not outcar_path.exists() or not poscar.exists() or not incar.exists():
        return None

    outcar_text = _read_memory_text(outcar_path, "OUTCAR")
    if not is_completed_text(outcar_text):
        return None

    n_atoms = _parse_memory_atom_count(poscar, _read_memory_text(poscar, "POSCAR"))
    ncore, kpar = _parse_memory_parallelism(incar, _read_memory_text(incar, "INCAR"))

    performance = parse_performance_evidence(outcar_text)
    if not performance.loop_times:
        raise VaspMemoryParseError(
            f"Completed OUTCAR has no valid electronic loop evidence: {outcar_path}"
        )
    total_ranks = performance.total_ranks
    nodes = max(1, total_ranks // gpus_per_node) if total_ranks > 0 else 1
    gpus = total_ranks if total_ranks > 0 else gpus_per_node
    return {
        "n_atoms": n_atoms,
        "n_kpoints_irr": performance.irreducible_kpoints,
        "n_electrons": int(performance.electrons),
        "nodes": nodes,
        "gpus": gpus,
        "ncore": ncore,
        "kpar": kpar,
        "avg_loop_time": f"{performance.average_loop_time:.4f}",
        "oom": 0,
    }


def _read_memory_text(path: Path, artifact: str) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise VaspMemoryParseError(f"Could not read required {artifact} artifact: {path}") from exc


def _parse_memory_atom_count(poscar: Path, text: str) -> int:
    poscar_lines = text.splitlines()
    if len(poscar_lines) <= 6:
        raise VaspMemoryParseError(f"POSCAR is missing its atom-count row: {poscar}")
    try:
        atom_counts = [int(value) for value in poscar_lines[6].split()]
    except (TypeError, ValueError) as exc:
        raise VaspMemoryParseError(f"POSCAR atom-count row is malformed: {poscar}") from exc
    if not atom_counts or any(value < 0 for value in atom_counts):
        raise VaspMemoryParseError(f"POSCAR atom-count row is malformed: {poscar}")
    return sum(atom_counts)


def _parse_memory_parallelism(incar: Path, text: str) -> tuple[int, int]:
    ncore = 0
    kpar = 0
    for line in text.splitlines():
        key_match = re.match(r"\s*(NCORE|KPAR)\b", line, re.IGNORECASE)
        if not key_match:
            continue
        match = re.match(r"\s*(NCORE|KPAR)\s*=\s*([^#!]+)", line, re.IGNORECASE)
        if not match:
            raise VaspMemoryParseError(
                f"{key_match.group(1).upper()} assignment is malformed: {incar}"
            )
        raw_value = match.group(2).strip()
        try:
            numeric_value = float(raw_value)
        except ValueError as exc:
            raise VaspMemoryParseError(
                f"{key_match.group(1).upper()} value is malformed: {incar}"
            ) from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise VaspMemoryParseError(f"{key_match.group(1).upper()} value is malformed: {incar}")
        value = int(numeric_value)
        if key_match.group(1).upper() == "NCORE":
            ncore = value
        else:
            kpar = value
    return ncore, kpar


__all__ = [
    "DftOutputValidationError",
    "ResolvedVaspOutput",
    "VaspJobEvidence",
    "VaspRegistryEvidence",
    "VASP_COMPLETION_MARKERS",
    "VaspParseResult",
    "VaspPerformanceEvidence",
    "VaspMemoryParseError",
    "outcar_is_complete",
    "is_completed_text",
    "parse_outcar",
    "parse_outcar_result",
    "parse_performance_evidence",
    "parse_memory_record",
    "parse_stress_from_outcar",
    "parse_virial_from_outcar",
    "resolve_verified_output",
    "validate_dft_result_labels",
]
