"""Concrete VASP adapter for the Phase 3 DFT backend boundary."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from ase.atoms import Atoms
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
from nepflow.io.atomic import atomic_write_bytes, atomic_write_text
from nepflow.io.json import write_json

from .failures import VaspFailureEvidence, classify_failure
from .inputs import (
    canonical_poscar_bytes,
    hash_incar_text,
    identity_for_structure,
)
from .outputs import (
    VASP_COMPLETION_MARKERS,
    ResolvedVaspOutput,
    VaspParseResult,
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
        incar_text: str | None = None,
    ) -> None:
        values = tuple(os.fspath(part) for part in command)
        if not values:
            raise ValueError("VASP command must not be empty")
        self._command = values
        self._input_files = {
            str(name).upper(): Path(path) for name, path in (input_files or {}).items()
        }
        if incar_text is not None and "INCAR" in self._input_files:
            raise ValueError("provide either input_files['INCAR'] or incar_text")
        self._incar_text = incar_text

    def _source_inputs(self, request: DftInputRequest) -> dict[str, Path]:
        source = Path(request.source_structure)
        root = source if source.is_dir() else source.parent
        paths: dict[str, Path] = {
            "POSCAR": self._input_files.get(
                "POSCAR", source / "POSCAR" if source.is_dir() else source
            ),
            "INCAR": self._input_files.get("INCAR", root / "INCAR"),
        }
        missing = [
            name
            for name in ("POSCAR", "INCAR")
            if (name != "INCAR" or self._incar_text is None) and not paths[name].is_file()
        ]
        if missing:
            raise BackendError("VASP required input artifact(s) missing: " + ", ".join(missing))
        return paths

    def _input_material(
        self,
        request: DftInputRequest,
    ) -> tuple[dict[str, Path], dict[str, bytes]]:
        paths = self._source_inputs(request)
        try:
            poscar_atoms = cast(
                Atoms,
                ase_read(
                    str(paths["POSCAR"]),
                    index=request.source_structure_index,
                ),
            )
        except Exception as exc:
            # ASE's VASP reader raises parser-specific exception classes.  At
            # this external boundary every such failure is a required-input
            # BackendError, never an empty or guessed structure.
            raise BackendError(f"Could not read required VASP artifact {paths['POSCAR']}") from exc
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
                raise BackendError(f"Could not read required VASP artifact {combined}") from exc
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
                raise BackendError(f"Could not read required VASP artifact {element_path}") from exc
        if missing:
            raise BackendError(
                "VASP required POTCAR artifact(s) missing for: " + ", ".join(missing)
            )
        return paths, potcar_data

    def _calculation_identity_and_material(
        self,
        request: DftInputRequest,
    ) -> tuple[DftCalculationIdentity, dict[str, Path], dict[str, bytes]]:
        paths, potcar_data = self._input_material(request)
        try:
            atoms = ase_read(
                str(paths["POSCAR"]),
                index=request.source_structure_index,
            )
        except Exception as exc:
            # Preserve parser failures as context-rich backend errors so a
            # malformed POSCAR cannot acquire a new identity by fallback.
            raise BackendError("Could not canonicalize VASP POSCAR") from exc
        if self._incar_text is None:
            try:
                incar_text = paths["INCAR"].read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise BackendError(
                    f"Could not read required VASP artifact {paths['INCAR']}"
                ) from exc
        else:
            incar_text = self._incar_text
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
        """Return the identity derived from structure, INCAR, and POTCAR inputs."""
        identity, _, _ = self._calculation_identity_and_material(request)
        return identity

    def prepare_inputs(self, request: DftInputRequest) -> DftInputArtifacts:
        """Canonicalize and atomically materialize the VASP input artifacts.

        The identity sidecar is written with the prepared inputs.  Missing or
        malformed source artifacts raise ``BackendError``; no structure or
        scientific parameter is guessed.
        """
        calculation, source_paths, potcar_data = self._calculation_identity_and_material(request)
        working = Path(request.working_directory)
        working.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for name in ("POSCAR", "INCAR"):
            source = source_paths[name]
            destination = working / name
            try:
                if name == "POSCAR":
                    atomic_write_text(
                        destination,
                        canonical_poscar_bytes(
                            ase_read(
                                str(source),
                                index=request.source_structure_index,
                            )
                        ).decode("utf-8"),
                        encoding="utf-8",
                    )
                elif name == "INCAR" and self._incar_text is not None:
                    atomic_write_text(destination, self._incar_text, encoding="utf-8")
                else:
                    atomic_write_bytes(destination, source.read_bytes())
            except (OSError, TypeError, ValueError) as exc:
                raise BackendError(
                    f"Could not prepare required VASP artifact {source}: {exc}"
                ) from exc
            files.append(destination)
        potcar_path = working / "POTCAR"
        try:
            atomic_write_bytes(
                potcar_path,
                b"".join(potcar_data[element] for element in sorted(potcar_data)),
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
        """Return the configured VASP command without shell syntax."""
        return self._command

    @property
    def executable_identity(self) -> tuple[str, ...]:
        """Return the configured executable identity for provenance."""

        return self._command

    def render_runner_script(self) -> str:
        """Render the backend runner body without scheduler directives."""
        bash_command = " ".join(self._command).replace("{ntasks}", "$TOTAL_RANKS")
        body = r"""
STRUCT_DIR="${1:-.}"
if [ -z "$STRUCT_DIR" ]; then
    echo "Usage: run_vasp.sh <struct_dir>"
    exit 1
fi

if [ ! -d "$STRUCT_DIR" ]; then
    echo "ERROR: Structure directory not found: $STRUCT_DIR"
    exit 1
fi

cd "$STRUCT_DIR" || exit 1

NCORE_VAL=$(grep -i "^[[:space:]]*NCORE" INCAR | grep "=" | head -1 | sed 's/.*=//;s/#.*//' | xargs)
KPAR_VAL=$(grep -i "^[[:space:]]*KPAR" INCAR | grep "=" | head -1 | sed 's/.*=//;s/#.*//' | xargs)
[ -z "$NCORE_VAL" ] && NCORE_VAL=2
[ -z "$KPAR_VAL" ] && KPAR_VAL=1

NNODES=${SLURM_NNODES:-1}
TOTAL_RANKS=${SLURM_NTASKS:-${SLURM_NNODES:-1}}

VASP_LOG="${STRUCT_DIR}/vasp_output.log"
rm -f "$STRUCT_DIR/.vasp_oom_detected"
printf '%s\n' "Resolved VASP command: __VASP_CMD__" > "$VASP_LOG"
printf '%s\n' "Resolved MPI ranks: $TOTAL_RANKS" >> "$VASP_LOG"
__VASP_CMD__ >> "$VASP_LOG" 2>&1
VASP_EXIT=$?
VASP_EXIT_MARKER="$STRUCT_DIR/.vasp_exit_code"
VASP_EXIT_MARKER_TMP="${VASP_EXIT_MARKER}.$$"
# Keep a numeric zero marker on success so every attempt has durable exit evidence.
printf '%s\n' "$VASP_EXIT" > "$VASP_EXIT_MARKER_TMP" && mv -f "$VASP_EXIT_MARKER_TMP" "$VASP_EXIT_MARKER"

if grep -Eiq "oom_kill|out[[:space:]_-]*of[[:space:]_-]*memory|killed" "$VASP_LOG" 2>/dev/null || [ "$VASP_EXIT" -eq 137 ]; then
    echo "1" > "$STRUCT_DIR/.vasp_oom_detected"
fi

if [ "$VASP_EXIT" -eq 0 ]; then
    rm -f CHG CHGCAR WAVECAR CONTCAR DOSCAR EIGENVAL PCDAT
fi
exit "$VASP_EXIT"
"""
        return body.replace("__VASP_CMD__", bash_command)

    def parse_completion(
        self,
        inputs: DftInputArtifacts,
        process=None,
    ) -> DftCompletionEvidence:
        """Read backend completion markers independently of scheduler state."""
        completed = outcar_is_complete(inputs.working_directory / "OUTCAR")
        markers = VASP_COMPLETION_MARKERS if completed else ()
        if (inputs.working_directory / ".vasp_oom_detected").is_file():
            markers = (*markers, "oom_kill")
        returncode = None if process is None else process.returncode
        if process is None:
            try:
                returncode = int(
                    (inputs.working_directory / ".vasp_exit_code")
                    .read_text(encoding="utf-8")
                    .strip()
                )
            except (OSError, ValueError):
                returncode = None
        return DftCompletionEvidence(
            completed=completed,
            markers=tuple(markers),
            returncode=returncode,
        )

    def parse_result(self, inputs: DftInputArtifacts) -> DftResult:
        """Parse a completed OUTCAR into verified eV-based DFT labels.

        Forces are ``(n_atoms, 3)`` eV/Angstrom and optional virial is a
        positive-compression Cartesian ``(3, 3)`` tensor in eV.  Missing,
        incomplete, or identity-mismatched output raises a typed backend error.
        """
        outcar = inputs.working_directory / "OUTCAR"
        if not outcar.is_file():
            raise BackendError(f"Required VASP output artifact is missing: {outcar}")
        if not outcar_is_complete(outcar):
            raise BackendError(f"VASP OUTCAR is incomplete: {outcar}")
        try:
            expected = cast(Atoms, ase_read(str(inputs.working_directory / "POSCAR")))
        except Exception as exc:
            # The prepared POSCAR is required evidence for output identity;
            # an ASE read failure must reject the completed output.
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
            raise BackendError(f"VASP output was rejected: {parsed.rejection_reason}")
        if parsed.structure_id != inputs.calculation.structure_id:
            raise ValidationError("VASP output structure identity does not match prepared inputs")
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
        """Classify combined scheduler, runner, and VASP evidence."""
        filesystem = (
            VaspFailureEvidence.from_job_directory(evidence.working_directory)
            if evidence.working_directory is not None
            else VaspFailureEvidence(
                job_directory=None,
                completed=False,
                oom_marker=False,
            )
        )
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
        runner_exit_code = (
            evidence.process.returncode
            if evidence.process is not None
            else filesystem.returncode
        )
        if runner_exit_code is None:
            runner_exit_code = returncode
        output_parts = tuple(
            value
            for value in (
                filesystem.output_log,
                process_output,
                "\n".join(evidence.markers),
            )
            if value
        )
        output_log = "\n".join(output_parts)
        output_lower = output_log.lower()
        runtime_output = "\n".join(
            line
            for line in output_log.splitlines()
            if not line.startswith("Resolved VASP command:")
            and not line.startswith("Resolved MPI ranks:")
        ).strip()
        scheduler_state = (evidence.scheduler_state or "").strip().lower()
        scheduler_reason = (evidence.scheduler_reason or "").strip()
        scheduler_reason_lower = scheduler_reason.lower()
        scheduler_oom = scheduler_state in {"oom", "out_of_memory"} or (
            "oom" in scheduler_reason_lower
            or "out of memory" in scheduler_reason_lower
            or "out_of_memory" in scheduler_reason_lower
        )
        scheduler_timeout = scheduler_state in {"timeout", "timed_out"} or (
            "timeout" in scheduler_reason_lower or "time limit" in scheduler_reason_lower
        )
        scheduler_cancelled = scheduler_state in {"cancelled", "canceled"}

        if evidence.completion.completed or filesystem.completed:
            return DftFailure(kind="completed", recoverable=False, reason="completed")
        if scheduler_timeout:
            return DftFailure(
                kind="timeout",
                recoverable=False,
                reason=scheduler_reason or "SLURM reported a timeout",
            )
        if scheduler_cancelled:
            return DftFailure(
                kind="cancelled",
                recoverable=False,
                reason=scheduler_reason or "SLURM cancelled the VASP job",
            )
        if (
            scheduler_oom
            or filesystem.oom_marker
            or "oom_kill" in output_lower
            or "out of memory" in output_lower
            or runner_exit_code == 137
            or any("oom" in marker.lower() for marker in evidence.markers)
        ):
            return classify_failure(
                VaspFailureEvidence(
                    job_directory=filesystem.job_directory,
                    completed=False,
                    oom_marker=True,
                    output_log=output_log,
                    returncode=runner_exit_code,
                )
            )
        if runner_exit_code not in (None, 0):
            return DftFailure(
                kind="vasp_execution_failed",
                recoverable=False,
                reason=f"VASP runner exited with return code {runner_exit_code}",
            )
        if runtime_output:
            return classify_failure(
                VaspFailureEvidence(
                    job_directory=filesystem.job_directory,
                    completed=False,
                    oom_marker=False,
                    output_log=runtime_output,
                    returncode=runner_exit_code,
                )
            )
        if (
            scheduler_state in {"failed", "node_fail", "boot_fail"}
            or evidence.scheduler_exit_code not in (None, 0)
        ):
            return DftFailure(
                kind="scheduler_execution_failed",
                recoverable=False,
                reason=(
                    scheduler_reason
                    or f"SLURM reported a failed VASP job (exit code {evidence.scheduler_exit_code})"
                ),
            )
        return DftFailure(
            kind="incomplete_output",
            recoverable=True,
            reason="VASP left no completion or failure evidence",
        )

    def validate_calculation_identity(
        self,
        expected: DftCalculationIdentity,
        observed: DftCalculationIdentity,
    ) -> None:
        """Raise ``ValidationError`` when scientific calculation IDs differ."""
        if expected.calculation_id != observed.calculation_id:
            raise ValidationError(
                "VASP calculation identity does not match expected scientific inputs"
            )


__all__ = ["VaspBackend"]
