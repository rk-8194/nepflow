"""Typed DFT preparation and reuse orchestration.

This module owns preparation policy only.  Scheduler submission and launcher
reconciliation remain separate dependencies for the following migration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import logging
from pathlib import Path
import shlex
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
    VaspRegistryEvidence,
    outcar_is_complete,
    resolve_verified_output,
)
from nepflow.dft.vasp.registry import (
    get_nepflow_root,
    get_registry_entry,
    read_completed_registry,
    read_status,
    write_status,
)
from nepflow.domain.identities import (
    DftCalculationIdentity,
    StructureIdentity,
    normalise_dft_calculation_identity,
)
from nepflow.errors import StateError
from nepflow.io.hashing import sha256_file
from nepflow.state import StateStore


logger = logging.getLogger(__name__)

_STALE_OUTPUTS = (
    "OUTCAR",
    "vasprun.xml",
    "OSZICAR",
    "vasp_output.log",
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
        expected.to_dict()
        if isinstance(expected, DftCalculationIdentity)
        else dict(expected)
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
        registry = read_completed_registry(get_nepflow_root(project_dir))
        prepared: list[PreparedCalculation] = []

        for dataset in datasets:
            source_path = selected_dir / f"{dataset}.xyz"
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Selected structures not found at {source_path}\n"
                    "Run the 'select' stage first."
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

                current_status = read_status(spec.working_directory)
                existing_identity = read_identity(spec.working_directory)
                current_match = bool(existing_identity) and calculation_identities_match(
                    calculation,
                    existing_identity,
                )
                current_evidence = None
                if current_match and current_status["status"] in {
                    "submitted",
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

                if not current_match or (
                    current_status["status"] not in {"submitted"}
                    and current_evidence is None
                ):
                    clean_stale_outputs(spec.working_directory)
                artifacts = backend.prepare_inputs(request)

                resolved = current_evidence or self._resolve_historical_output(
                    state_store,
                    registry,
                    calculation,
                )
                if current_match and current_status["status"] == "submitted" and resolved is None:
                    status = "submitted"
                elif resolved is not None:
                    status = "reused" if resolved.verification_source != "current_job_identity" else "completed"
                else:
                    status = "pending"

                write_status(
                    spec.working_directory,
                    status=status,
                    retry_level=current_status.get("retry_level", 0)
                    if current_match
                    else 0,
                    structure_id=calculation.structure_id,
                    calculation_id=calculation.calculation_id,
                    incar_hash=calculation.incar_hash,
                    potcar_hash=calculation.potcar_hash,
                    **(
                        {
                            "reused_from": str(resolved.outcar_path.parent.resolve())
                        }
                        if resolved is not None
                        and resolved.verification_source != "current_job_identity"
                        else {}
                    ),
                )
                prepared.append(
                    PreparedCalculation(
                        spec=spec,
                        artifacts=artifacts,
                        status=status,
                        reused_output=resolved,
                    )
                )

        self._persist(state_store, prepared)
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
            path.name: path
            for path in vasp_config_dir.glob("POTCAR_*")
            if path.is_file()
        }
        return VaspBackend(
            command=command,
            input_files=input_files,
            incar_text=inject_incar_defaults(
                incar_template.read_text(encoding="utf-8"),
                config,
            ),
        )

    @staticmethod
    def _resolve_historical_output(
        state_store: StateStore,
        registry: Mapping[str, object],
        calculation: DftCalculationIdentity,
    ) -> ResolvedVaspOutput | None:
        state_result = VaspPreparationOrchestrator._resolve_state_artifact(
            state_store,
            calculation,
        )
        if state_result is not None:
            return state_result

        entry = get_registry_entry(
            dict(registry),
            calculation.incar_hash,
            calculation.potcar_hash,
            calculation.structure_id,
        )
        if entry is None:
            return None
        job_path = Path(entry["job_path"])
        outcar = job_path / "OUTCAR"
        expected_hash = entry.get("outcar_hash")
        observed_hash = sha256_file(outcar, required=False)
        if expected_hash is not None and expected_hash != observed_hash:
            return None
        evidence_identity = {
            key: entry.get(key, calculation.to_dict()[key])
            for key in ("structure_id", "incar_hash", "potcar_hash", "calculation_id")
        }
        return resolve_verified_output(
            calculation.to_dict(),
            registry_evidence=VaspRegistryEvidence(outcar, evidence_identity),
        )

    @staticmethod
    def _resolve_state_artifact(
        state_store: StateStore,
        calculation: DftCalculationIdentity,
    ) -> ResolvedVaspOutput | None:
        row = state_store.get_dft_calculation(calculation.calculation_id)
        if row is None or row.get("status") != "completed":
            return None
        attempt_id = row.get("accepted_attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            return None
        for artifact in state_store.list_artifacts(originating_attempt_id=attempt_id):
            if artifact.get("artifact_type") != "vasp_outcar":
                continue
            path_value = artifact.get("path")
            if not isinstance(path_value, str) or not path_value.strip():
                continue
            outcar = Path(path_value)
            if not outcar_is_complete(outcar):
                continue
            if artifact.get("sha256") != sha256_file(outcar):
                continue
            return resolve_verified_output(
                calculation.to_dict(),
                registry_evidence=VaspRegistryEvidence(outcar, calculation.to_dict()),
            )
        return None

    @staticmethod
    def _persist(
        state_store: StateStore,
        prepared: Sequence[PreparedCalculation],
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
                state_store.upsert_dft_calculation(
                    calculation,
                    status="completed" if item.reused_output is not None else "prepared",
                    selected=True,
                    reused_from_calculation_id=(
                        calculation.calculation_id if item.reused else None
                    ),
                    metadata={
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
                attempts = state_store.list_dft_attempts(calculation.calculation_id)
                attempt_id = (
                    str(attempts[0]["attempt_id"])
                    if attempts
                    else f"{calculation.calculation_id}:attempt:1"
                )
                if not attempts:
                    state_store.create_dft_attempt(
                        calculation.calculation_id,
                        attempt_id,
                        attempt_number=1,
                        status="completed" if item.reused_output is not None else item.status,
                        metadata={"working_directory": str(item.spec.working_directory)},
                    )
                if item.reused_output is not None:
                    from nepflow.domain.identities import ArtifactIdentity

                    artifact = ArtifactIdentity.from_file(
                        "vasp_outcar",
                        item.reused_output.outcar_path,
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
