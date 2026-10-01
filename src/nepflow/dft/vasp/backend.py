"""Concrete VASP adapter for the Phase 3 DFT backend boundary."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from ase.io import read as ase_read

from nepflow.dft.backend import (
    DftCompletionEvidence,
    DftFailure,
    DftFailureEvidence,
    DftInputArtifacts,
    DftInputRequest,
    DftResult,
)
from nepflow.domain.calculations import DftResultArtifact
from nepflow.domain.identities import (
    ArtifactIdentity,
    DftCalculationIdentity,
    StructureIdentity,
)
from nepflow.errors import BackendError, ValidationError
from nepflow.io.hashing import sha256_bytes, sha256_file
from nepflow.io.json import write_json

from .failures import VaspFailureEvidence, classify_failure
from .inputs import canonical_poscar_bytes, hash_incar_text, hash_potcar_bytes
from .outputs import (
    ResolvedVaspOutput,
    VaspParseResult,
    VASP_COMPLETION_MARKERS,
    outcar_is_complete,
    parse_outcar_result,
)


class VaspBackend:
    """VASP implementation of the generic DFT protocol.

    ``source_structure`` is the source POSCAR (or an ASE-readable structure
    file).  INCAR and POTCAR are required beside it unless ``input_files``
    supplies explicit paths.  KPOINTS is optional, but when present its hash
    participates in scientific identity.
    """

    def __init__(
        self,
        command: Sequence[str | os.PathLike[str]] = ("vasp_std",),
        *,
        input_files: Mapping[str, str | os.PathLike[str]] | None = None,
    ) -> None:
        values = tuple(os.fspath(part) for part in command)
        if not values:
            raise ValueError("VASP command must not be empty")
        self._command = values
        self._input_files = {
            str(name).upper(): Path(path) for name, path in (input_files or {}).items()
        }

    def _source_inputs(self, request: DftInputRequest) -> dict[str, Path]:
        source = Path(request.source_structure)
        root = source if source.is_dir() else source.parent
        paths = {
            "POSCAR": self._input_files.get(
                "POSCAR", source / "POSCAR" if source.is_dir() else source
            ),
            "INCAR": self._input_files.get("INCAR", root / "INCAR"),
            "POTCAR": self._input_files.get("POTCAR", root / "POTCAR"),
        }
        kpoints = self._input_files.get("KPOINTS", root / "KPOINTS")
        if kpoints.exists():
            paths["KPOINTS"] = kpoints
        missing = [name for name in ("POSCAR", "INCAR", "POTCAR") if not paths[name].is_file()]
        if missing:
            raise BackendError(
                "VASP required input artifact(s) missing: " + ", ".join(missing)
            )
        return paths

    def _identity_payload(self, request: DftInputRequest) -> dict[str, str]:
        paths = self._source_inputs(request)
        incar_hash = hash_incar_text(paths["INCAR"].read_text(encoding="utf-8"))
        potcar_hash = hash_potcar_bytes(paths["POTCAR"].read_bytes())
        payload = {
            "structure_id": request.structure.structure_id,
            "incar_hash": incar_hash,
            "potcar_hash": potcar_hash,
        }
        try:
            canonical_poscar = canonical_poscar_bytes(ase_read(str(paths["POSCAR"])))
        except Exception as exc:
            raise BackendError("Could not canonicalize VASP POSCAR") from exc
        poscar_hash = sha256_bytes(canonical_poscar)
        payload["poscar_hash"] = poscar_hash
        if "KPOINTS" in paths:
            payload["kpoints_hash"] = sha256_file(paths["KPOINTS"])
        return payload

    def calculation_identity(self, request: DftInputRequest) -> DftCalculationIdentity:
        payload = self._identity_payload(request)
        return DftCalculationIdentity(**payload)

    def prepare_inputs(self, request: DftInputRequest) -> DftInputArtifacts:
        calculation = self.calculation_identity(request)
        source_paths = self._source_inputs(request)
        working = Path(request.working_directory)
        working.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for name, source in source_paths.items():
            destination = working / name
            try:
                if name == "POSCAR":
                    destination.write_text(
                        canonical_poscar_bytes(ase_read(str(source))).decode("utf-8"),
                        encoding="utf-8",
                    )
                else:
                    shutil.copyfile(source, destination)
            except (OSError, TypeError, ValueError) as exc:
                raise BackendError(
                    f"Could not prepare required VASP artifact {source}: {exc}"
                ) from exc
            files.append(destination)
        write_json(
            working / ".vasp_identity",
            {
                "structure_id": calculation.structure_id,
                "incar_hash": calculation.incar_hash,
                "potcar_hash": calculation.potcar_hash,
                "calculation_id": calculation.calculation_id,
            },
        )
        return DftInputArtifacts(
            calculation=calculation,
            working_directory=working,
            files=tuple(files),
            requirements=request.requirements,
        )

    def execution_command(self, inputs: DftInputArtifacts) -> tuple[str, ...]:
        return self._command

    def parse_completion(
        self,
        inputs: DftInputArtifacts,
        process=None,
    ) -> DftCompletionEvidence:
        completed = outcar_is_complete(inputs.working_directory / "OUTCAR")
        markers = VASP_COMPLETION_MARKERS if completed else ()
        return DftCompletionEvidence(
            completed=completed,
            markers=tuple(markers),
            returncode=None if process is None else process.returncode,
        )

    def parse_result(self, inputs: DftInputArtifacts) -> DftResult:
        outcar = inputs.working_directory / "OUTCAR"
        if not outcar.is_file():
            raise BackendError(f"Required VASP output artifact is missing: {outcar}")
        try:
            expected = ase_read(str(inputs.working_directory / "POSCAR"))
        except Exception as exc:
            raise BackendError("Could not read prepared VASP POSCAR") from exc
        identity = {
            "structure_id": inputs.calculation.structure_id,
            "incar_hash": inputs.calculation.incar_hash,
            "potcar_hash": inputs.calculation.potcar_hash,
            "calculation_id": inputs.calculation.calculation_id,
        }
        evidence = ResolvedVaspOutput(
            outcar_path=outcar,
            calculation_identity=tuple(sorted(identity.items())),
            verification_source="prepared_input_identity",
        )
        parsed: VaspParseResult = parse_outcar_result(
            outcar,
            expected,
            require_virial=inputs.requirements.virial_requested,
            calculation_identity=identity,
            identity_evidence=evidence,
        )
        if not parsed.accepted:
            raise BackendError(
                f"VASP output was rejected: {parsed.rejection_reason}"
            )
        if parsed.structure_id != inputs.calculation.structure_id:
            raise ValidationError(
                "VASP output structure identity does not match prepared inputs"
            )
        artifact = DftResultArtifact(
            calculation=inputs.calculation,
            outcar=ArtifactIdentity.from_file("vasp_outcar", outcar),
            status="completed",
        )
        return DftResult(
            structure=StructureIdentity(inputs.calculation.structure_id),
            calculation=inputs.calculation,
            energy_ev=parsed.energy_ev,
            forces_ev_per_angstrom=parsed.forces_ev_per_angstrom,
            virial_ev=parsed.virial_ev,
            artifact=artifact,
            requirements=inputs.requirements,
        )

    def classify_failure(self, evidence: DftFailureEvidence) -> DftFailure:
        return classify_failure(
            VaspFailureEvidence(
                job_directory=Path(evidence.calculation.structure_id),
                completed=evidence.completion.completed,
                oom_marker=any("oom" in marker.lower() for marker in evidence.markers),
                output_log=" ".join(evidence.markers),
                returncode=None if evidence.process is None else evidence.process.returncode,
            )
        )

    def validate_calculation_identity(
        self,
        expected: DftCalculationIdentity,
        observed: DftCalculationIdentity,
    ) -> None:
        if expected.calculation_id != observed.calculation_id:
            raise ValidationError(
                "VASP calculation identity does not match expected scientific inputs"
            )


__all__ = ["VaspBackend"]
