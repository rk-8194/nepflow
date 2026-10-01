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
from nepflow.io.json import write_json

from .failures import VaspFailureEvidence, classify_failure
from .inputs import (
    canonical_poscar_bytes,
    hash_incar_text,
    identity_for_structure,
)
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
    supplies explicit paths.  The current Phase 2 identity contract consists
    of structure, scientific INCAR, and POTCAR identity only.
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
        paths: dict[str, Path] = {
            "POSCAR": self._input_files.get(
                "POSCAR", source / "POSCAR" if source.is_dir() else source
            ),
            "INCAR": self._input_files.get("INCAR", root / "INCAR"),
        }
        missing = [name for name in ("POSCAR", "INCAR") if not paths[name].is_file()]
        if missing:
            raise BackendError(
                "VASP required input artifact(s) missing: " + ", ".join(missing)
            )
        return paths

    def _input_material(
        self,
        request: DftInputRequest,
    ) -> tuple[dict[str, Path], dict[str, bytes]]:
        paths = self._source_inputs(request)
        try:
            poscar_atoms = ase_read(str(paths["POSCAR"]))
        except Exception as exc:
            raise BackendError(
                f"Could not read required VASP artifact {paths['POSCAR']}"
            ) from exc
        root = Path(request.source_structure)
        root = root if root.is_dir() else root.parent
        unique_elements = sorted(set(poscar_atoms.get_chemical_symbols()))
        combined = self._input_files.get("POTCAR", root / "POTCAR")
        potcar_data: dict[str, bytes] = {}
        if combined.is_file():
            if len(unique_elements) != 1:
                raise BackendError(
                    "A combined POTCAR is supported only for single-element VASP inputs"
                )
            try:
                potcar_data[unique_elements[0]] = combined.read_bytes()
            except (OSError, UnicodeError) as exc:
                raise BackendError(
                    f"Could not read required VASP artifact {combined}"
                ) from exc
            paths["POTCAR"] = combined
            return paths, potcar_data

        missing: list[str] = []
        for element in unique_elements:
            element_path = self._input_files.get(
                f"POTCAR_{element.upper()}",
                root / f"POTCAR_{element}",
            )
            if not element_path.is_file():
                missing.append(element)
                continue
            try:
                potcar_data[element] = element_path.read_bytes()
            except (OSError, UnicodeError) as exc:
                raise BackendError(
                    f"Could not read required VASP artifact {element_path}"
                ) from exc
        if missing:
            raise BackendError(
                "VASP required POTCAR artifact(s) missing for: "
                + ", ".join(missing)
            )
        return paths, potcar_data

    def _calculation_identity_and_material(
        self,
        request: DftInputRequest,
    ) -> tuple[DftCalculationIdentity, dict[str, Path], dict[str, bytes]]:
        paths, potcar_data = self._input_material(request)
        try:
            atoms = ase_read(str(paths["POSCAR"]))
        except Exception as exc:
            raise BackendError("Could not canonicalize VASP POSCAR") from exc
        try:
            incar_text = paths["INCAR"].read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise BackendError(
                f"Could not read required VASP artifact {paths['INCAR']}"
            ) from exc
        incar_hash = hash_incar_text(incar_text)
        identity = identity_for_structure(
            atoms,
            {"incar_hash": incar_hash, "potcar_data": potcar_data},
        ).calculation
        if identity.structure_id != request.structure.structure_id:
            raise ValidationError(
                "VASP source structure does not match requested structure identity"
            )
        return identity, paths, potcar_data

    def calculation_identity(self, request: DftInputRequest) -> DftCalculationIdentity:
        identity, _, _ = self._calculation_identity_and_material(request)
        return identity

    def prepare_inputs(self, request: DftInputRequest) -> DftInputArtifacts:
        calculation, source_paths, potcar_data = self._calculation_identity_and_material(request)
        working = Path(request.working_directory)
        working.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for name in ("POSCAR", "INCAR"):
            source = source_paths[name]
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
        potcar_path = working / "POTCAR"
        try:
            potcar_path.write_bytes(
                b"".join(potcar_data[element] for element in sorted(potcar_data))
            )
        except OSError as exc:
            raise BackendError(f"Could not prepare required VASP artifact POTCAR: {exc}") from exc
        files.append(potcar_path)
        try:
            write_json(
                working / ".vasp_identity",
                {
                    **calculation.scientific_payload(),
                    "calculation_id": calculation.calculation_id,
                },
            )
        except (OSError, TypeError, ValueError) as exc:
            raise BackendError(
                f"Could not persist required VASP identity sidecar in {working}"
            ) from exc
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
        if not outcar_is_complete(outcar):
            raise BackendError(f"VASP OUTCAR is incomplete: {outcar}")
        try:
            expected = ase_read(str(inputs.working_directory / "POSCAR"))
        except Exception as exc:
            raise BackendError("Could not read prepared VASP POSCAR") from exc
        identity = inputs.calculation.to_dict()
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
        process_output = "\n".join(
            value
            for value in (
                evidence.process.stdout if evidence.process is not None else None,
                evidence.process.stderr if evidence.process is not None else None,
            )
            if value
        )
        returncode = (
            evidence.completion.returncode
            if evidence.process is None
            else evidence.process.returncode
        )
        output_lower = process_output.lower()
        output_parts = tuple(evidence.markers) + ((process_output,) if process_output else ())
        return classify_failure(
            VaspFailureEvidence(
                job_directory=None,
                completed=evidence.completion.completed,
                oom_marker=(
                    any("oom" in marker.lower() for marker in evidence.markers)
                    or "oom" in output_lower
                    or returncode == 137
                ),
                output_log="\n".join(output_parts),
                returncode=returncode,
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
