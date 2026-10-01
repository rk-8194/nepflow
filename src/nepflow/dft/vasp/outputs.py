"""Canonical VASP output completion, parsing, and performance evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable, Mapping

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


def parse_virial_from_outcar(outcar_path: Path, volume: float) -> np.ndarray | None:
    """Parse VASP stress and convert it to NEPFlow's virial convention."""
    try:
        outcar_text = Path(outcar_path).read_text(
            encoding="utf-8", errors="replace"
        )
        stress_pattern = (
            r"STRESS\s+in cartesian coordinates \(kB\)\n"
            r"\s+([-+.\d]+)\s+([-+.\d]+)\s+([-+.\d]+)\n"
            r"\s+([-+.\d]+)\s+([-+.\d]+)\s+([-+.\d]+)\n"
            r"\s+([-+.\d]+)\s+([-+.\d]+)\s+([-+.\d]+)"
        )
        matches = list(re.finditer(stress_pattern, outcar_text))
        if not matches:
            return None
        values = [float(matches[-1].group(index)) for index in range(1, 10)]
        stress = np.asarray(values, dtype=float).reshape(3, 3)
        return virial_from_stress(
            stress_kbar_to_ev_per_angstrom3(stress), float(volume)
        )
    except (OSError, TypeError, ValueError):
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
    reader = ase_read if reader is None else reader
    identity_reader = read_identity if identity_reader is None else identity_reader
    structure_id = calculate_structure_id(ase_atoms)
    identity_items_source = dict(calculation_identity or {"structure_id": structure_id})
    comparison_identity = normalise_dft_calculation_identity(identity_items_source)
    identity_items = tuple(
        sorted((str(key), str(value)) for key, value in identity_items_source.items())
    )
    comparison_items = tuple(
        sorted((str(key), str(value)) for key, value in comparison_identity.items())
    )
    source_outcar = str(Path(outcar_path).resolve())
    source_hash = sha256_file(Path(outcar_path), required=False)

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
        if identity_evidence is None:
            source_identity = identity_reader(Path(outcar_path).parent)
            if not source_identity:
                return rejected("missing_calculation_identity")
            if any(
                source_identity.get(key) != value
                for key, value in comparison_identity.items()
            ):
                return rejected("incompatible_calculation_identity")
        elif (
            identity_evidence.outcar_path.resolve() != Path(outcar_path).resolve()
            or identity_evidence.calculation_identity != comparison_items
        ):
            return rejected("incompatible_calculation_identity")

    try:
        atoms = reader(str(outcar_path))
    except Exception as exc:
        return rejected(f"outcar_parse_failed:{type(exc).__name__}:{exc}")

    expected_species = tuple(ase_atoms.get_chemical_symbols())
    actual_species = tuple(atoms.get_chemical_symbols())
    if len(atoms) != len(ase_atoms):
        return rejected(
            f"atom_count_mismatch:expected={len(ase_atoms)}:actual={len(atoms)}"
        )
    if actual_species != expected_species:
        return rejected(
            f"species_mismatch:expected={expected_species}:actual={actual_species}"
        )

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
        validate_dft_result_labels(structure)
    except DftOutputValidationError as exc:
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
    """Parse the performance evidence used by the memory benchmark."""
    ranks_match = re.search(r"running on\s+(\d+)\s+total cores", outcar_text)
    mpi_ranks_match = re.search(r"running\s+(\d+)\s+mpi-ranks", outcar_text)
    loops = tuple(
        float(match.group(1))
        for match in re.finditer(
            r"LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)",
            outcar_text,
        )
    )
    kpoints_match = re.search(
        r"Found\s+(\d+)\s+irreducible k-points", outcar_text
    )
    electrons_match = re.search(r"NELECT\s*=\s*([\d.]+)", outcar_text)
    return VaspPerformanceEvidence(
        total_ranks=int(ranks_match.group(1)) if ranks_match else 0,
        mpi_ranks=int(mpi_ranks_match.group(1)) if mpi_ranks_match else 0,
        loop_times=loops,
        irreducible_kpoints=int(kpoints_match.group(1)) if kpoints_match else 0,
        electrons=float(electrons_match.group(1)) if electrons_match else 0.0,
    )


__all__ = [
    "DftOutputValidationError",
    "ResolvedVaspOutput",
    "VASP_COMPLETION_MARKERS",
    "VaspParseResult",
    "VaspPerformanceEvidence",
    "outcar_is_complete",
    "is_completed_text",
    "parse_outcar",
    "parse_outcar_result",
    "parse_performance_evidence",
    "parse_virial_from_outcar",
    "validate_dft_result_labels",
]
