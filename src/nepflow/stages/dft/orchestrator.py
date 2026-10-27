"""Typed DFT preparation and reuse orchestration.

Preparation remains separate from the execution reconciler: this module
creates/verifies backend inputs, while ``reconciliation.py`` advances their
StateStore-backed attempts through scheduler and backend evidence.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ase.io import iread

from nepflow.config.models import NepflowConfig
from nepflow.dft.backend import (
    DftBackend,
    DftInputArtifacts,
    DftInputRequest,
    DftResultRequirements,
)
from nepflow.dft.vasp.backend import VaspBackend
from nepflow.dft.vasp.inputs import (
    VaspInputIdentity,
    inject_incar_defaults,
    read_identity,
)
from nepflow.dft.vasp.outputs import (
    ResolvedVaspOutput,
    VaspJobEvidence,
    outcar_is_complete,
    resolve_verified_output,
)
from nepflow.domain.identities import (
    DftCalculationIdentity,
    StructureIdentity,
    normalise_dft_calculation_identity,
)
from nepflow.errors import StateError
from nepflow.io.hashing import sha256_file
from nepflow.state import StateStore
from nepflow.state._dft_records import dft_attempt_id

logger = logging.getLogger(__name__)

_STALE_OUTPUTS = (
    "OUTCAR",
    "vasprun.xml",
    "OSZICAR",
    "vasp_output.log",
    ".vasp_exit_code",
    ".vasp_oom_detected",
    "CHG",
    "CHGCAR",
    "WAVECAR",
    "CONTCAR",
    "DOSCAR",
    "EIGENVAL",
    "PCDAT",
)


@dataclass(frozen=True, slots=True)
class DftCalculationSpec:
    """One selected structure's typed preparation request."""

    project_name: str
    dataset: str
    selected_index: int
    source_structure: Path
    working_directory: Path
    structure: StructureIdentity
    requirements: DftResultRequirements = field(default_factory=DftResultRequirements)

    def request(self) -> DftInputRequest:
        return DftInputRequest(
            structure=self.structure,
            source_structure=self.source_structure,
            working_directory=self.working_directory,
            source_structure_index=self.selected_index,
            requirements=self.requirements,
        )


@dataclass(frozen=True, slots=True)
class PreparedCalculation:
    """Prepared backend inputs plus verified reuse evidence, if any."""

    spec: DftCalculationSpec
    artifacts: DftInputArtifacts
    status: str
    reused_output: ResolvedVaspOutput | None = None

    @property
    def calculation(self) -> DftCalculationIdentity:
        return self.artifacts.calculation

    @property
    def vasp_identity(self) -> VaspInputIdentity:
        """Expose the backend's canonical VASP scientific identity."""
        return VaspInputIdentity(self.calculation)

    @property
    def reused(self) -> bool:
        return self.status == "reused"


@dataclass(frozen=True, slots=True)
class DftPreparationResult:
    """Typed result of one preparation pass."""

    calculations: tuple[PreparedCalculation, ...]

    @property
    def prepared_count(self) -> int:
        return len(self.calculations)

    @property
    def reused_count(self) -> int:
        return sum(calculation.reused for calculation in self.calculations)


class DftPreparationOrchestrator(Protocol):
    """Dependency boundary consumed by :class:`DftStage`."""

    def prepare_calculations(
        self,
        *,
        config: NepflowConfig,
        project_dir: Path,
        project_name: str,
        state_store: StateStore,
        datasets: Sequence[str],
    ) -> DftPreparationResult: ...


def calculation_identities_match(
    expected: DftCalculationIdentity | Mapping[str, object],
    observed: Mapping[str, object],
) -> bool:
    """Compare the complete scientific identity, including calculation ID."""
    expected_payload = (
        expected.to_dict() if isinstance(expected, DftCalculationIdentity) else dict(expected)
    )
    expected_payload = normalise_dft_calculation_identity(expected_payload)
    observed_payload = normalise_dft_calculation_identity(observed)
    return bool(expected_payload) and all(
        observed_payload.get(key) == expected_payload.get(key)
        for key in ("structure_id", "incar_hash", "potcar_hash", "calculation_id")
    )


def clean_stale_outputs(working_directory: Path) -> None:
    """Remove outputs that belong to a prior scientific identity."""
    for filename in _STALE_OUTPUTS:
        (Path(working_directory) / filename).unlink(missing_ok=True)


def _scientific_attempts(
    state_store: StateStore,
    calculation_id: str,
    *,
    execution_generation: int | None = None,
) -> list[dict[str, object]]:
    """Exclude explicitly namespaced benchmark observations from DFT resume."""

    if execution_generation is None:
        calculation = state_store.get_dft_calculation(calculation_id)
        metadata = {} if calculation is None else calculation.get("metadata", {})
        try:
            execution_generation = (
                int(metadata.get("dft_execution_generation", 0))
                if isinstance(metadata, Mapping)
                else 0
            )
        except (TypeError, ValueError):
            execution_generation = 0

    return [
        attempt
        for attempt in state_store.list_dft_attempts(calculation_id)
        if not _is_benchmark_attempt(attempt)
        and _attempt_generation(attempt) == execution_generation
    ]


def _is_benchmark_attempt(attempt: Mapping[str, object]) -> bool:
    metadata = attempt.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("execution_kind") == "vasp_benchmark"


def _attempt_number(attempt: Mapping[str, object]) -> int:
    value = attempt.get("attempt_number", 1)
    return int(value) if isinstance(value, (int, float, str)) else 1


def _attempt_generation(attempt: Mapping[str, object]) -> int:
    metadata = attempt.get("metadata")
    if not isinstance(metadata, Mapping):
        return 0
    try:
        return int(metadata.get("dft_execution_generation", 0))
    except (TypeError, ValueError):
        return 0


def _attempt_retry_level(attempt: Mapping[str, object]) -> int:
    metadata = attempt.get("metadata")
    if isinstance(metadata, Mapping) and "retry_level" in metadata:
        try:
            return max(0, int(metadata["retry_level"]))
        except (TypeError, ValueError):
            pass
    return max(0, _attempt_number(attempt) - 1)


def _calculation_generation(state_store: StateStore, calculation_id: str) -> int:
    calculation = state_store.get_dft_calculation(calculation_id)
    metadata = {} if calculation is None else calculation.get("metadata", {})
    try:
        return (
            int(metadata.get("dft_execution_generation", 0))
            if isinstance(metadata, Mapping)
            else 0
        )
    except (TypeError, ValueError):
        return 0


def _active_attempt_for_working_directory(
    state_store: StateStore,
    working_directory: Path,
) -> dict[str, object] | None:
    """Find an authoritative active attempt owning a prepared directory."""

    target = Path(working_directory).resolve()
    # Inspect attempt rows as well as calculation rows.  A controller may have
    # persisted the attempt transition before the calculation summary update;
    # the attempt is still authoritative for whether this directory is live.
    for calculation in state_store.list_dft_calculations():
        calculation_id = calculation.get("calculation_id")
        if not isinstance(calculation_id, str):
            continue
        calculation_metadata = calculation.get("metadata")
        try:
            generation = (
                int(calculation_metadata.get("dft_execution_generation", 0))
                if isinstance(calculation_metadata, Mapping)
                else 0
            )
        except (TypeError, ValueError):
            generation = 0
        for attempt in state_store.list_dft_attempts(calculation_id):
            if _is_benchmark_attempt(attempt):
                continue
            if attempt.get("status") not in {"submitted", "running"}:
                continue
            if _attempt_generation(attempt) != generation:
                continue
            metadata = attempt.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            value = metadata.get("working_directory")
            if isinstance(value, str) and Path(value).resolve() == target:
                return attempt
    return None


class VaspPreparationOrchestrator:
    """Prepare VASP inputs and record typed calculation/reuse state."""

    def __init__(
        self,
        *,
        backend: DftBackend | None = None,
        scheduler: object | None = None,
        requirements: DftResultRequirements | None = None,
    ) -> None:
        self.backend = backend
        self.scheduler = scheduler
        self.requirements = requirements or DftResultRequirements()

    def prepare_calculations(
        self,
        *,
        config: NepflowConfig,
        project_dir: Path,
        project_name: str,
        state_store: StateStore,
        datasets: Sequence[str] = ("train", "test"),
    ) -> DftPreparationResult:
        if not isinstance(config, NepflowConfig):
            raise TypeError("DFT preparation requires typed NepflowConfig")
        if not isinstance(state_store, StateStore):
            raise TypeError("DFT preparation requires StateStore")

        project_dir = Path(project_dir)
        selected_dir = project_dir / "structures" / "selected"
        jobs_dir = project_dir / "vasp" / "jobs"
        backend = self.backend or self._build_backend(config, project_dir)
        prepared: list[PreparedCalculation] = []

        for dataset in datasets:
            source_path = selected_dir / f"{dataset}.xyz"
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Selected structures not found at {source_path}\nRun the 'select' stage first."
                )
            for selected_index, atoms in enumerate(iread(str(source_path), format="extxyz")):
                spec = DftCalculationSpec(
                    project_name=project_name,
                    dataset=dataset,
                    selected_index=selected_index,
                    source_structure=source_path,
                    working_directory=jobs_dir / dataset / f"struct_{selected_index:04d}",
                    structure=StructureIdentity.from_atoms(atoms),
                    requirements=self.requirements,
                )
                request = spec.request()
                calculation = backend.calculation_identity(request)
                if calculation.structure_id != spec.structure.structure_id:
                    raise StateError(
                        "DFT backend returned a structure identity different from the selected structure"
                    )

                existing_identity = read_identity(spec.working_directory)
                current_match = bool(existing_identity) and calculation_identities_match(
                    calculation,
                    existing_identity,
                )
                attempts = _scientific_attempts(state_store, calculation.calculation_id)
                latest_attempt = attempts[-1] if attempts else None
                persisted_calculation = state_store.get_dft_calculation(
                    calculation.calculation_id
                )
                persisted_status = (
                    None
                    if latest_attempt is None
                    else str(latest_attempt.get("status"))
                )
                if persisted_status is None and persisted_calculation is not None:
                    raw_status = persisted_calculation.get("status")
                    persisted_status = str(raw_status) if raw_status else None
                persisted_status = (
                    "pending"
                    if persisted_status == "prepared"
                    else persisted_status
                )
                current_status = {
                    "status": persisted_status or "pending",
                    "retry_level": (
                        _attempt_retry_level(latest_attempt)
                        if latest_attempt is not None
                        else 0
                    ),
                }
                active_status = current_status["status"] in {"submitted", "running"}
                protected_status = active_status or current_status["status"] in {
                    "completed",
                    "reused",
                }
                if not current_match:
                    active_owner = _active_attempt_for_working_directory(
                        state_store,
                        spec.working_directory,
                    )
                    if active_owner is not None:
                        raise StateError(
                            "Cannot replace prepared VASP inputs owned by an active attempt: "
                            f"{spec.working_directory}"
                        )
                if protected_status and not current_match:
                    raise StateError(
                        "Persisted DFT attempt has no matching prepared VASP identity sidecar: "
                        f"{spec.working_directory}"
                    )

                artifacts: DftInputArtifacts | None = None
                if current_match:
                    loader = getattr(backend, "load_prepared_inputs", None)
                    if not callable(loader):
                        if protected_status:
                            raise StateError(
                                "Backend cannot validate existing prepared VASP inputs for "
                                f"{spec.working_directory}"
                            )
                    else:
                        try:
                            artifacts = loader(
                                request,
                                expected_calculation=calculation,
                            )
                        except StateError:
                            if protected_status:
                                raise

                current_evidence = None
                if current_match and current_status["status"] in {
                    "submitted",
                    "running",
                    "completed",
                    "reused",
                }:
                    current_evidence = resolve_verified_output(
                        calculation.to_dict(),
                        current_jobs=(
                            VaspJobEvidence(
                                spec.working_directory,
                                existing_identity,
                                current_status,
                            ),
                        ),
                    )

                if artifacts is None:
                    if not current_match:
                        clean_stale_outputs(spec.working_directory)
                    artifacts = backend.prepare_inputs(request)

                resolved = current_evidence or self._resolve_historical_output(
                    state_store,
                    calculation,
                    canonical_path=spec.working_directory / "OUTCAR",
                    project_dir=project_dir,
                )
                if (
                    current_status["status"] in {"completed", "reused"}
                    and resolved is None
                ):
                    raise StateError(
                        "Persisted completed DFT calculation has no verified output: "
                        f"{calculation.calculation_id}"
                    )
                if current_match and persisted_status == "failed":
                    status = "failed"
                elif (
                    current_match
                    and persisted_status in {"completed", "reused"}
                    and resolved is None
                ):
                    status = persisted_status
                elif (
                    current_match
                    and current_status["status"] in {"submitted", "running"}
                    and resolved is None
                ):
                    status = current_status["status"]
                elif resolved is not None:
                    status = (
                        "reused"
                        if resolved.verification_source != "current_job_identity"
                        else "completed"
                    )
                else:
                    status = "pending"

                prepared.append(
                    PreparedCalculation(
                        spec=spec,
                        artifacts=artifacts,
                        status=status,
                        reused_output=resolved,
                    )
                )

        self._persist(state_store, prepared, project_root=project_dir)
        logger.info(
            "Prepared %d DFT calculations (%d reused)",
            len(prepared),
            sum(item.reused for item in prepared),
        )
        return DftPreparationResult(tuple(prepared))

    @staticmethod
    def _build_backend(config: NepflowConfig, project_dir: Path) -> DftBackend:
        command = tuple(shlex.split(config.hpc.vasp_command))
        if not command:
            raise ValueError("Required configuration hpc.vasp_command must not be blank")
        vasp_config_dir = project_dir / "config" / "vasp"
        incar_template = vasp_config_dir / "INCAR"
        if not incar_template.is_file():
            raise FileNotFoundError(
                f"INCAR template not found at {incar_template}\n"
                f"Place your VASP INCAR file in: {vasp_config_dir}/"
            )
        input_files = {
            path.name: path for path in vasp_config_dir.glob("POTCAR_*") if path.is_file()
        }
        return VaspBackend(
            command=command,
            input_files=input_files,
            incar_text=inject_incar_defaults(
                incar_template.read_text(encoding="utf-8"),
                config.vasp,
            ),
        )

    @staticmethod
    def _resolve_historical_output(
        state_store: StateStore,
        calculation: DftCalculationIdentity,
        *,
        canonical_path: Path | None = None,
        project_dir: Path | None = None,
    ) -> ResolvedVaspOutput | None:
        return VaspPreparationOrchestrator._resolve_state_artifact(
            state_store,
            calculation,
            canonical_path=canonical_path,
            project_dir=project_dir,
        )

    @staticmethod
    def _resolve_state_artifact(
        state_store: StateStore,
        calculation: DftCalculationIdentity,
        *,
        canonical_path: Path | None = None,
        project_dir: Path | None = None,
    ) -> ResolvedVaspOutput | None:
        row = state_store.get_dft_calculation(calculation.calculation_id)
        if row is None or row.get("status") != "completed":
            return None
        generation = _calculation_generation(state_store, calculation.calculation_id)
        attempt_id = row.get("accepted_attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            return None
        accepted_attempt = state_store.get_dft_attempt(attempt_id)
        if accepted_attempt is None or _attempt_generation(accepted_attempt) != generation:
            return None
        for artifact in state_store.list_artifacts(originating_attempt_id=attempt_id):
            if artifact.get("artifact_type") != "vasp_outcar":
                continue
            path_value = artifact.get("path")
            if not isinstance(path_value, str) or not path_value.strip():
                continue
            if canonical_path is not None:
                outcar = Path(canonical_path).resolve()
            elif project_dir is not None:
                try:
                    outcar = state_store.resolve_artifact_path(
                        artifact,
                        project_dir,
                        allow_legacy_absolute=True,
                    )
                except StateError:
                    continue
            else:
                outcar = Path(path_value)
            if not outcar_is_complete(outcar):
                continue
            if artifact.get("sha256") != sha256_file(outcar):
                continue
            return ResolvedVaspOutput(
                outcar_path=outcar,
                calculation_identity=tuple(
                    sorted((str(key), str(value)) for key, value in calculation.to_dict().items())
                ),
                verification_source="state_store_artifact",
            )
        return None

    @staticmethod
    def _persist(
        state_store: StateStore,
        prepared: Sequence[PreparedCalculation],
        *,
        project_root: Path | None = None,
    ) -> None:
        with state_store.transaction():
            for item in prepared:
                calculation = item.calculation
                state_store.upsert_structure(
                    item.spec.structure,
                    metadata={
                        "source": str(item.spec.source_structure.resolve()),
                        "dataset": item.spec.dataset,
                        "selected_index": item.spec.selected_index,
                    },
                )
                persisted_calculation = state_store.get_dft_calculation(
                    calculation.calculation_id
                )
                persisted_metadata = (
                    dict(persisted_calculation.get("metadata", {}))
                    if isinstance(persisted_calculation, Mapping)
                    and isinstance(persisted_calculation.get("metadata"), Mapping)
                    else {}
                )
                generation = _calculation_generation(state_store, calculation.calculation_id)
                state_store.upsert_dft_calculation(
                    calculation,
                    status=(
                        "completed"
                        if item.reused_output is not None
                        else item.status
                    ),
                    selected=True,
                    reused_from_calculation_id=(
                        calculation.calculation_id if item.reused else None
                    ),
                    metadata={
                        **persisted_metadata,
                        "project_name": item.spec.project_name,
                        "dataset": item.spec.dataset,
                        "selected_index": item.spec.selected_index,
                        "preparation_status": item.status,
                        "reuse_verification": (
                            item.reused_output.verification_source
                            if item.reused_output is not None
                            else None
                        ),
                    },
                )
                attempts = _scientific_attempts(state_store, calculation.calculation_id)
                attempt_number = (
                    _attempt_number(attempts[-1])
                    if attempts
                    else state_store.allocate_dft_attempt_number(calculation.calculation_id)
                )
                attempt_id = (
                    str(attempts[-1]["attempt_id"])
                    if attempts
                    else dft_attempt_id(calculation.calculation_id, attempt_number, generation)
                )
                if not attempts:
                    state_store.create_dft_attempt(
                        calculation.calculation_id,
                        attempt_id,
                        attempt_number=attempt_number,
                        status="completed" if item.reused_output is not None else item.status,
                        metadata={
                            "working_directory": str(item.spec.working_directory),
                            "dft_execution_generation": generation,
                            "retry_level": 0,
                        },
                    )
                if item.reused_output is not None:
                    from nepflow.domain.identities import ArtifactIdentity

                    artifact = ArtifactIdentity.from_file(
                        "vasp_outcar",
                        item.reused_output.outcar_path,
                        project_root=project_root,
                    )
                    state_store.register_completed_result(
                        calculation.calculation_id,
                        attempt_id,
                        [artifact],
                    )


def prepare_calculations(
    *,
    config: NepflowConfig,
    project_dir: Path,
    project_name: str,
    state_store: StateStore,
    backend: DftBackend | None = None,
    scheduler: object | None = None,
    datasets: Sequence[str] = ("train", "test"),
) -> DftPreparationResult:
    """Prepare calculations through the canonical VASP orchestration seam."""
    return VaspPreparationOrchestrator(
        backend=backend,
        scheduler=scheduler,
    ).prepare_calculations(
        config=config,
        project_dir=project_dir,
        project_name=project_name,
        state_store=state_store,
        datasets=datasets,
    )


__all__ = [
    "DftCalculationSpec",
    "DftPreparationOrchestrator",
    "DftPreparationResult",
    "PreparedCalculation",
    "VaspPreparationOrchestrator",
    "calculation_identities_match",
    "clean_stale_outputs",
    "prepare_calculations",
]
