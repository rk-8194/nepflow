"""Concrete GPUMD adapter for the static-prediction protocol.

The adapter owns only the scientific GPUMD input and output contract.  It
does not submit jobs or infer model/dataset identities from directory names.
Callers provide an explicit :class:`StaticPredictionRequest`; output parsing
then returns the actual values serialized by GPUMD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import shutil
from typing import Sequence

import numpy as np
from ase.atoms import Atoms
from ase.io import read as ase_read

from nepflow.errors import MlipError
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
            "GPUMD output forces must have shape "
            f"({len(atoms)}, 3), got {forces.shape}"
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
            stress = np.array(
                [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float
            )
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
        input_path = Path(request.input_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"GPUMD model input not found: {input_path}")
        working_directory = Path(request.working_directory)
        working_directory.mkdir(parents=True, exist_ok=True)
        working_input_path = working_directory / "model.xyz"
        if input_path.resolve() != working_input_path.resolve():
            shutil.copy2(input_path, working_input_path)
        model_artifact_path = request.model.artifact.model.path
        if model_artifact_path is not None:
            source_potential = Path(model_artifact_path)
            if source_potential.is_file():
                target_potential = working_directory / "nep.txt"
                if source_potential.resolve() != target_potential.resolve():
                    shutil.copy2(source_potential, target_potential)
        run_in_path = working_directory / "run.in"
        output_path = working_directory / "out.xyz"
        content = self.render_input(
            replicates=replicates,
            potential_filename=potential_filename,
        )
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
        output = Path(output_path) if output_path is not None else Path(request.working_directory) / "out.xyz"
        atoms = _last_frame(output)
        if len(atoms) != request.atom_count:
            raise MlipError(
                "GPUMD output atom count does not match the static prediction request: "
                f"expected {request.atom_count}, got {len(atoms)}"
            )
        virial = _virial(atoms, required=request.virial_requested)
        return StaticPrediction(
            structure=request.structure,
            model_run=request.model.identity,
            atom_count=len(atoms),
            energy_ev=_finite_energy(atoms),
            forces_ev_per_angstrom=_forces(atoms),
            virial_ev=virial,
            virial_requested=request.virial_requested,
            runtime=PredictionRuntimeMetadata(backend_version="gpumd"),
            species=tuple(atoms.get_chemical_symbols()),
            positions_angstrom=np.asarray(atoms.positions, dtype=float),
            cell_angstrom=np.asarray(atoms.cell, dtype=float),
            pbc=tuple(bool(value) for value in atoms.pbc),
        )

    # Keep the protocol spelling as the primary public operation.  Execution
    # is intentionally outside this adapter; predict parses completed output.
    def predict(self, request: StaticPredictionRequest) -> StaticPrediction:
        return self.parse_prediction(request)

    parse_output = parse_prediction


__all__ = ["GpumdBackend", "GpumdStaticInput"]
