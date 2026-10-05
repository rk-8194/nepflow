"""Concrete GPUMD adapter for the static-prediction protocol.

The adapter owns only the scientific GPUMD input and output contract.  It
does not submit jobs or infer model/dataset identities from directory names.
Callers provide an explicit :class:`StaticPredictionRequest`; output parsing
then returns the actual values serialized by GPUMD.
"""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ase.atoms import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from nepflow.errors import MlipError
from nepflow.io.hashing import sha256_file
from nepflow.mlip.simulation import (
    PredictionRuntimeMetadata,
    StaticPrediction,
    StaticPredictionRequest,
)


@dataclass(frozen=True, slots=True)
class GpumdStaticInput:
    """Files and protocol text for one prepared GPUMD static evaluation."""

    input_path: Path
    run_in_path: Path
    output_path: Path
    content: str


def _finite_energy(atoms: Atoms) -> float:
    try:
        value = float(atoms.get_potential_energy())
    except Exception as exc:  # ASE raises several calculator-specific errors.
        raise MlipError("GPUMD output is missing a readable total energy") from exc
    if not np.isfinite(value):
        raise MlipError("GPUMD output energy is not finite")
    return value


def _forces(atoms: Atoms) -> np.ndarray:
    value = atoms.arrays.get("force")
    if value is None:
        value = atoms.arrays.get("forces")
    if value is None:
        raise MlipError("GPUMD output is missing per-atom forces")
    forces = np.asarray(value, dtype=float)
    if forces.shape != (len(atoms), 3):
        raise MlipError(
            f"GPUMD output forces must have shape ({len(atoms)}, 3), got {forces.shape}"
        )
    if not np.isfinite(forces).all():
        raise MlipError("GPUMD output forces contain non-finite values")
    return forces


def _tensor(value: object, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MlipError(f"GPUMD output {label} is not numeric") from exc
    if result.shape == (9,):
        result = result.reshape(3, 3)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise MlipError(f"GPUMD output {label} must be a finite 3x3 tensor")
    return result


def _virial(atoms: Atoms, *, required: bool) -> np.ndarray | None:
    if "virial" in atoms.info:
        return _tensor(atoms.info["virial"], "virial")
    if "stress" in atoms.info:
        # This is the ASE/extended-XYZ stress convention.  GPUMD's static
        # protocol consumes the positive-compression virial in eV.
        stress = np.asarray(atoms.info["stress"], dtype=float)
        if stress.shape == (6,):
            xx, yy, zz, yz, xz, xy = stress
            stress = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)
        stress = _tensor(stress, "stress")
        volume = float(atoms.get_volume())
        if not np.isfinite(volume) or volume <= 0:
            raise MlipError("GPUMD output stress requires a positive cell volume")
        return -stress * volume
    if required:
        raise MlipError("GPUMD output is missing requested virial/stress")
    return None


def _last_frame(path: Path) -> Atoms:
    if not Path(path).is_file():
        raise FileNotFoundError(f"GPUMD output not found: {path}")
    try:
        frames = ase_read(str(path), index=":", format="extxyz")
    except Exception as exc:
        # ASE's extxyz reader is an external parser boundary; all parser
        # failures become a typed MLIP error and no frame is substituted.
        raise MlipError(f"Could not parse GPUMD output {path}: {exc}") from exc
    if isinstance(frames, Atoms):
        return frames
    if not frames:
        raise MlipError(f"GPUMD output contains no frames: {path}")
    return frames[-1]


class GpumdBackend:
    """Parse and render the existing one-step GPUMD static protocol."""

    def __init__(self, command: str | Sequence[str] = ("gpumd",)) -> None:
        if isinstance(command, str):
            command = tuple(shlex.split(command))
        self._command = tuple(str(part) for part in command)
        if not self._command or any(not part for part in self._command):
            raise ValueError("GPUMD command must contain executable arguments")

    @property
    def command(self) -> tuple[str, ...]:
        return self._command

    @property
    def executable_identity(self) -> tuple[str, ...]:
        return self._command

    @staticmethod
    def render_input(
        *,
        replicates: tuple[int, int, int] = (1, 1, 1),
        potential_filename: str = "nep.txt",
        output_filename: str = "out.xyz",
    ) -> str:
        values = tuple(int(value) for value in replicates)
        if len(values) != 3 or any(value < 1 for value in values):
            raise ValueError("GPUMD replicate factors must be three positive integers")
        if not potential_filename or Path(potential_filename).name != potential_filename:
            raise ValueError("GPUMD potential filename must be a local filename")
        if not output_filename or Path(output_filename).name != output_filename:
            raise ValueError("GPUMD output filename must be a local filename")
        return "\n".join(
            (
                f"replicate {values[0]} {values[1]} {values[2]}",
                f"potential {potential_filename}",
                "ensemble nve",
                "time_step 0",
                f"dump_xyz 1 {output_filename} precision double force potential virial",
                "run 1",
                "",
            )
        )

    def prepare_inputs(
        self,
        request: StaticPredictionRequest,
        *,
        replicates: tuple[int, int, int] = (1, 1, 1),
        potential_filename: str = "nep.txt",
    ) -> GpumdStaticInput:
        artifact = request.model.artifact
        if artifact is None:
            raise MlipError("GPUMD input preparation requires a model artifact")
        model_artifact_path = artifact.model.path
        if not model_artifact_path:
            raise MlipError("GPUMD model artifact is missing its source path")
        source_potential = Path(model_artifact_path)
        if not source_potential.is_file():
            raise MlipError(f"GPUMD model artifact does not exist: {source_potential}")
        actual_hash = sha256_file(source_potential, error_type=MlipError)
        if actual_hash != artifact.model.sha256:
            raise MlipError(
                "GPUMD model artifact SHA-256 does not match the authoritative identity"
            )
        content = self.render_input(
            replicates=replicates,
            potential_filename=potential_filename,
        )
        input_path = Path(request.input_path)
        working_directory = Path(request.working_directory)
        working_directory.mkdir(parents=True, exist_ok=True)
        working_input_path = working_directory / "model.xyz"
        if request.species is not None and request.positions_angstrom is not None:
            if request.cell_angstrom is None or request.pbc is None:
                raise MlipError(
                    "GPUMD input materialization requires cell and periodic-boundary metadata"
                )
            atoms = Atoms(
                symbols=list(request.species),
                positions=np.asarray(request.positions_angstrom, dtype=float),
                cell=np.asarray(request.cell_angstrom, dtype=float),
                pbc=request.pbc,
            )
            ase_write(
                str(working_input_path),
                atoms,
                format="extxyz",
                write_info=False,
                write_results=False,
            )
        elif not input_path.is_file():
            raise FileNotFoundError(f"GPUMD model input not found: {input_path}")
        elif input_path.resolve() != working_input_path.resolve():
            shutil.copy2(input_path, working_input_path)
        target_potential = working_directory / potential_filename
        if source_potential.resolve() != target_potential.resolve():
            shutil.copy2(source_potential, target_potential)
        run_in_path = working_directory / "run.in"
        output_path = working_directory / "out.xyz"
        run_in_path.write_text(content, encoding="utf-8", newline="\n")
        return GpumdStaticInput(working_input_path, run_in_path, output_path, content)

    # A descriptive alias for callers that use backend terminology rather
    # than the DFT-style prepare_inputs spelling.
    prepare_static_input = prepare_inputs

    def parse_prediction(
        self,
        request: StaticPredictionRequest,
        output_path: Path | None = None,
    ) -> StaticPrediction:
        output = (
            Path(output_path)
            if output_path is not None
            else Path(request.working_directory) / "out.xyz"
        )
        atoms = _last_frame(output)
        if len(atoms) != request.atom_count:
            raise MlipError(
                "GPUMD output atom count does not match the static prediction request: "
                f"expected {request.atom_count}, got {len(atoms)}"
            )
        atom_mapping = self._validate_configuration(request, atoms)
        virial = _virial(atoms, required=request.virial_requested)
        return StaticPrediction(
            structure=request.structure,
            model_run=request.model.identity,
            atom_count=len(atoms),
            energy_ev=_finite_energy(atoms),
            forces_ev_per_angstrom=_forces(atoms),
            virial_ev=virial,
            virial_requested=request.virial_requested,
            runtime=PredictionRuntimeMetadata(
                backend_version="gpumd",
                command=self._command,
            ),
            species=tuple(atoms.get_chemical_symbols()),
            positions_angstrom=np.asarray(atoms.positions, dtype=float),
            cell_angstrom=np.asarray(atoms.cell, dtype=float),
            pbc=(bool(atoms.pbc[0]), bool(atoms.pbc[1]), bool(atoms.pbc[2])),
            atom_mapping=atom_mapping,
        )

    @staticmethod
    def _validate_configuration(
        request: StaticPredictionRequest,
        atoms: Atoms,
    ) -> tuple[int, ...] | None:
        """Validate output geometry and return output-order reference mapping."""

        expected_species = request.expected_species
        expected_positions = request.expected_positions_angstrom
        expected_cell = request.expected_cell_angstrom
        expected_pbc = request.expected_pbc
        if any(
            value is not None
            for value in (expected_species, expected_positions, expected_cell, expected_pbc)
        ) and not all(
            value is not None
            for value in (expected_species, expected_positions, expected_cell, expected_pbc)
        ):
            raise MlipError("GPUMD request has incomplete expected physical configuration")
        if expected_species is None:
            return None
        actual_species = tuple(atoms.get_chemical_symbols())
        if actual_species == expected_species:
            species_indices = list(range(len(atoms)))
        else:
            species_indices = []
        if expected_cell is None or expected_positions is None or expected_pbc is None:
            raise MlipError("GPUMD request is missing expected geometry metadata")
        actual_cell = np.asarray(atoms.cell, dtype=float)
        if not np.allclose(actual_cell, expected_cell, rtol=0.0, atol=1.0e-7):
            raise MlipError("GPUMD output cell does not match the replicated case cell")
        if tuple(bool(value) for value in atoms.pbc) != tuple(expected_pbc):
            raise MlipError("GPUMD output periodic-boundary flags do not match the case")
        if not species_indices:
            remaining = list(range(len(atoms)))
            species_indices = []
            for expected_symbol in expected_species:
                candidates = [
                    index for index in remaining if actual_species[index] == expected_symbol
                ]
                if len(candidates) != 1:
                    raise MlipError(
                        "GPUMD output species do not match the replicated case configuration"
                    )
                species_indices.append(candidates[0])
                remaining.remove(candidates[0])
        if len(species_indices) != len(expected_species):
            raise MlipError("GPUMD output species count does not match the case configuration")

        try:
            inverse_cell = np.linalg.inv(expected_cell)
        except np.linalg.LinAlgError as exc:
            raise MlipError("GPUMD expected case cell is singular") from exc
        actual_positions = np.asarray(atoms.positions, dtype=float)
        expected_fractional = np.asarray(expected_positions) @ inverse_cell
        actual_fractional = actual_positions @ inverse_cell
        used: set[int] = set()
        output_to_expected: list[int] = []
        for actual_index, actual_symbol in enumerate(actual_species):
            candidates = [
                expected_index
                for expected_index, expected_symbol in enumerate(expected_species)
                if expected_index not in used and expected_symbol == actual_symbol
            ]
            if not candidates:
                raise MlipError("GPUMD output contains an unexpected species")
            distances: list[tuple[float, int]] = []
            for expected_index in candidates:
                delta = actual_fractional[actual_index] - expected_fractional[expected_index]
                delta = np.asarray(delta, dtype=float)
                for axis, periodic in enumerate(expected_pbc):
                    if periodic:
                        delta[axis] -= np.rint(delta[axis])
                cartesian_delta = delta @ expected_cell
                distances.append((float(np.linalg.norm(cartesian_delta)), expected_index))
            distance, expected_index = min(distances)
            if not np.isfinite(distance) or distance > 1.0e-6:
                raise MlipError(
                    "GPUMD output positions do not match the replicated case configuration"
                )
            used.add(expected_index)
            output_to_expected.append(expected_index)
        if len(used) != len(expected_species):
            raise MlipError("GPUMD output is missing a replicated case atom")
        if request.atom_mapping is None:
            return tuple(output_to_expected)
        if len(request.atom_mapping) != len(expected_species):
            raise MlipError("GPUMD request atom_mapping does not match expected geometry")
        return tuple(request.atom_mapping[index] for index in output_to_expected)

    # Keep the protocol spelling as the primary public operation.  Execution
    # is intentionally outside this adapter; predict parses completed output.
    def predict(self, request: StaticPredictionRequest) -> StaticPrediction:
        return self.parse_prediction(request)

    parse_output = parse_prediction


__all__ = ["GpumdBackend", "GpumdStaticInput"]
