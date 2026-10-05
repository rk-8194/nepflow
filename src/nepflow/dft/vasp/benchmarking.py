"""Typed VASP benchmark planning, execution, and provenance.

Benchmarking is an explicit DFT service.  It owns the description of a
resource observation, while input preparation, scheduler reconciliation, and
VASP interpretation remain owned by their canonical boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nepflow.config.models import NepflowConfig
from nepflow.dft.backend import DftBackend, DftInputArtifacts, DftInputRequest
from nepflow.dft.vasp.inputs import set_incar_parameters
from nepflow.dft.vasp.outputs import VaspPerformanceEvidence, parse_performance_evidence
from nepflow.domain.identities import StructureIdentity
from nepflow.errors import StateError
from nepflow.hpc.resources import JobResources
from nepflow.hpc.scheduler import Scheduler
from nepflow.io.hashing import sha256_canonical_json

if TYPE_CHECKING:
    from nepflow.stages.dft.reconciliation import (
        DftExecutionRecord,
        DftReconciliationResult,
    )


class BenchmarkOutcome(str, Enum):
    """Outcome vocabulary for one benchmark observation."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    OUT_OF_MEMORY = "out_of_memory"
    FAILED = "failed"


_DEFAULT_BENCHMARK_PARAMETERS: tuple[tuple[str, object], ...] = (
    ("NELM", 10),
    ("NSW", 0),
)


@dataclass(frozen=True, slots=True)
class VaspBenchmarkTarget:
    """One explicitly declared structure/compatibility target."""

    structure: StructureIdentity
    source_structure: Path
    compatibility_key: str
    label: str = ""
    source_structure_index: int = 0

    def __post_init__(self) -> None:
        if not self.compatibility_key.strip():
            raise ValueError("benchmark compatibility_key must not be blank")
        if self.source_structure_index < 0:
            raise ValueError("source_structure_index must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure_id": self.structure.structure_id,
            "source_structure": str(self.source_structure),
            "source_structure_index": self.source_structure_index,
            "compatibility_key": self.compatibility_key,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class VaspBenchmarkResource:
    """One explicit VASP resource/INCAR combination."""

    resources: JobResources
    ncore: int
    kpar: int

    def __post_init__(self) -> None:
        if self.ncore < 1 or self.kpar < 1:
            raise ValueError("ncore and kpar must be positive")
        if self.resources.gpus_per_node < 1:
            raise ValueError("VASP benchmark resources require at least one GPU per node")

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": asdict(self.resources),
            "ncore": self.ncore,
            "kpar": self.kpar,
        }


def _powers_of_two(limit: int) -> tuple[int, ...]:
    values: list[int] = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= 2
    return tuple(values)


def build_benchmark_resources(
    *,
    cores_per_node: int,
    gpus_per_node: int,
    nodes: int = 1,
    walltime: str | None = None,
    memory_per_node: int | str | None = None,
    ncore_values: Sequence[int] | None = None,
    gpu_values: Sequence[int] | None = None,
    kpar_values: Sequence[int] | None = None,
) -> tuple[VaspBenchmarkResource, ...]:
    """Build a deterministic grid using typed, unambiguous GPU semantics."""

    if cores_per_node < 1 or gpus_per_node < 1:
        raise ValueError("cores_per_node and gpus_per_node must be positive")
    ncores = tuple(
        ncore_values
        or (value for value in _powers_of_two(cores_per_node) if cores_per_node % value == 0)
    )
    if not ncores:
        ncores = (cores_per_node,)
    gpu_counts_per_node = tuple(gpu_values or _powers_of_two(gpus_per_node))
    if not gpu_counts_per_node:
        gpu_counts_per_node = (gpus_per_node,)
    kpars = tuple(kpar_values or _powers_of_two(gpus_per_node))

    result: list[VaspBenchmarkResource] = []
    for gpus_per_node_for_case in sorted(set(gpu_counts_per_node)):
        if gpus_per_node_for_case < 1 or gpus_per_node_for_case > gpus_per_node:
            raise ValueError("each benchmark GPU count must fit on one node")
        for ncore in sorted(set(ncores)):
            if ncore < 1 or ncore > cores_per_node:
                raise ValueError("each benchmark NCORE value must fit on one node")
            for kpar in sorted(set(kpars)):
                if kpar < 1 or kpar > nodes * gpus_per_node_for_case:
                    continue
                resources = JobResources(
                    nodes=nodes,
                    gpus_per_node=gpus_per_node_for_case,
                    mpi_ranks=nodes * gpus_per_node_for_case,
                    memory_per_node=memory_per_node,
                    walltime=walltime,
                )
                result.append(VaspBenchmarkResource(resources, ncore, kpar))
    if not result:
        raise ValueError("benchmark resource grid is empty")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class VaspBenchmarkCase:
    """One planned benchmark execution."""

    benchmark_id: str
    target: VaspBenchmarkTarget
    resource: VaspBenchmarkResource
    working_directory: Path
    benchmark_parameters: tuple[tuple[str, object], ...] = _DEFAULT_BENCHMARK_PARAMETERS

    @property
    def job_name(self) -> str:
        return f"nfbench_{self.benchmark_id[-16:]}"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            key: value
            for key, value in (
                *self.benchmark_parameters,
                ("NCORE", self.resource.ncore),
                ("KPAR", self.resource.kpar),
            )
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "target": self.target.to_dict(),
            "resource": self.resource.to_dict(),
            "working_directory": str(self.working_directory),
            "benchmark_parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class VaspBenchmarkPlan:
    """Deterministic benchmark plan with no implicit material assumption."""

    project_name: str
    root_directory: Path
    cases: tuple[VaspBenchmarkCase, ...]
    site: str | None = None
    hardware: str | None = None
    executable_identity: str | None = None
    benchmark_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("benchmark plan must contain at least one case")
        payload = {
            "project_name": self.project_name,
            "site": self.site,
            "hardware": self.hardware,
            "executable_identity": self.executable_identity,
            "cases": [case.to_dict() for case in self.cases],
        }
        object.__setattr__(
            self,
            "benchmark_id",
            "vasp_benchmark_" + sha256_canonical_json(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "project_name": self.project_name,
            "root_directory": str(self.root_directory),
            "site": self.site,
            "hardware": self.hardware,
            "executable_identity": self.executable_identity,
            "cases": [case.to_dict() for case in self.cases],
        }


def build_vasp_benchmark_plan(
    *,
    project_name: str,
    root_directory: Path,
    targets: Sequence[VaspBenchmarkTarget],
    resources: Sequence[VaspBenchmarkResource],
    site: str | None = None,
    hardware: str | None = None,
    executable_identity: str | None = None,
    benchmark_parameters: Mapping[str, object] | None = None,
) -> VaspBenchmarkPlan:
    """Build cases from explicit structures and resources in stable order."""

    ordered_targets = tuple(
        sorted(
            targets,
            key=lambda item: (item.compatibility_key, item.structure.structure_id),
        )
    )
    ordered_resources = tuple(
        sorted(
            resources,
            key=lambda item: (
                item.resources.nodes,
                item.resources.gpus_per_node,
                item.resources.mpi_ranks,
                item.ncore,
                item.kpar,
            ),
        )
    )
    if not ordered_targets or not ordered_resources:
        raise ValueError("benchmark plan requires targets and resources")
    root = Path(root_directory)
    effective_parameters = _benchmark_parameters(benchmark_parameters)
    cases = [
        _build_benchmark_case(root, target, resource, effective_parameters)
        for target in ordered_targets
        for resource in ordered_resources
    ]
    return VaspBenchmarkPlan(
        project_name=project_name,
        root_directory=root,
        cases=tuple(cases),
        site=site,
        hardware=hardware,
        executable_identity=executable_identity,
    )


def _benchmark_parameters(
    benchmark_parameters: Mapping[str, object] | None,
) -> tuple[tuple[str, object], ...]:
    extra_parameters = tuple(
        sorted(
            ((str(key).upper(), value) for key, value in (benchmark_parameters or {}).items()),
            key=lambda item: item[0],
        )
    )
    return extra_parameters or _DEFAULT_BENCHMARK_PARAMETERS


def _build_benchmark_case(
    root: Path,
    target: VaspBenchmarkTarget,
    resource: VaspBenchmarkResource,
    benchmark_parameters: tuple[tuple[str, object], ...],
) -> VaspBenchmarkCase:
    target_label = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        target.label or target.structure.structure_id[:12],
    )
    case_payload = {
        "target": target.to_dict(),
        "resource": resource.to_dict(),
        "benchmark_parameters": dict(benchmark_parameters),
    }
    case_id = "case_" + sha256_canonical_json(case_payload)
    case_dir = (
        root
        / target_label
        / (
            f"nodes{resource.resources.nodes}_"
            f"gpus{resource.resources.gpus_per_node}_"
            f"ranks{resource.resources.mpi_ranks}_"
            f"ncore{resource.ncore}_kpar{resource.kpar}"
        )
    )
    return VaspBenchmarkCase(
        benchmark_id=case_id,
        target=target,
        resource=resource,
        working_directory=case_dir,
        benchmark_parameters=benchmark_parameters,
    )


def build_vasp_benchmark_plan_from_config(
    config: NepflowConfig,
    *,
    project_name: str,
    root_directory: Path,
    targets: Sequence[VaspBenchmarkTarget],
    site: str | None = None,
    hardware: str | None = None,
    executable_identity: str | None = None,
    memory_per_node: int | str | None = None,
) -> VaspBenchmarkPlan:
    """Build an explicit benchmark plan from canonical project resources."""

    resources = build_benchmark_resources(
        cores_per_node=config.hpc.cores_per_node,
        gpus_per_node=config.hpc.gpus_per_node,
        walltime=config.dft_recovery.vasp_walltime,
        memory_per_node=memory_per_node,
    )
    return build_vasp_benchmark_plan(
        project_name=project_name,
        root_directory=root_directory,
        targets=targets,
        resources=resources,
        site=site,
        hardware=hardware,
        executable_identity=executable_identity,
    )


@dataclass(frozen=True, slots=True)
class VaspBenchmarkProvenance:
    """Compatibility identity for a benchmark observation."""

    compatibility_key: str
    structure_id: str
    calculation_id: str
    site: str | None
    hardware: str | None
    executable_identity: str | None
    resources: JobResources

    def compatible_with(self, other: "VaspBenchmarkProvenance") -> bool:
        return (
            self.compatibility_key == other.compatibility_key
            and self.structure_id == other.structure_id
            and self.calculation_id == other.calculation_id
            and self.site == other.site
            and self.hardware == other.hardware
            and self.executable_identity == other.executable_identity
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatibility_key": self.compatibility_key,
            "structure_id": self.structure_id,
            "calculation_id": self.calculation_id,
            "site": self.site,
            "hardware": self.hardware,
            "executable_identity": self.executable_identity,
            "resources": asdict(self.resources),
        }


@dataclass(frozen=True, slots=True)
class VaspBenchmarkResult:
    """Structured benchmark result; failed observations are never successful."""

    benchmark_id: str
    attempt_id: str
    outcome: BenchmarkOutcome
    provenance: VaspBenchmarkProvenance
    performance: VaspPerformanceEvidence | None = None
    failure_kind: str | None = None
    failure_reason: str | None = None

    @property
    def successful(self) -> bool:
        return self.outcome is BenchmarkOutcome.COMPLETED and self.performance is not None

    def can_reuse_for(self, provenance: VaspBenchmarkProvenance) -> bool:
        return self.successful and self.provenance.compatible_with(provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "attempt_id": self.attempt_id,
            "outcome": self.outcome.value,
            "provenance": self.provenance.to_dict(),
            "performance": (
                None
                if self.performance is None
                else {
                    "total_ranks": self.performance.total_ranks,
                    "mpi_ranks": self.performance.mpi_ranks,
                    "loop_times": list(self.performance.loop_times),
                    "irreducible_kpoints": self.performance.irreducible_kpoints,
                    "electrons": self.performance.electrons,
                }
            ),
            "failure_kind": self.failure_kind,
            "failure_reason": self.failure_reason,
        }


def compatible_benchmark_observations(
    results: Sequence[VaspBenchmarkResult],
    provenance: VaspBenchmarkProvenance,
) -> tuple[VaspBenchmarkResult, ...]:
    """Return only successful observations with matching compatibility data."""

    return tuple(result for result in results if result.can_reuse_for(provenance))


@dataclass(frozen=True, slots=True)
class VaspBenchmarkExecution:
    case: VaspBenchmarkCase
    inputs: DftInputArtifacts
    record: DftExecutionRecord
    provenance: VaspBenchmarkProvenance


@dataclass(frozen=True, slots=True)
class VaspBenchmarkPass:
    execution: DftReconciliationResult
    results: tuple[VaspBenchmarkResult, ...]


class _BenchmarkPerformanceRecorder:
    """Adapt canonical backend evidence into benchmark events."""

    def __init__(self, state_store: Any, plan: VaspBenchmarkPlan) -> None:
        self.state_store = state_store
        self.plan = plan
        self.results: dict[str, VaspBenchmarkResult] = {}

    def record_completion(self, record: DftExecutionRecord, result: Any) -> None:
        del result
        provenance = _provenance_from_record(record)
        outcar = record.inputs.working_directory / "OUTCAR"
        try:
            performance = parse_performance_evidence(
                outcar.read_text(encoding="utf-8", errors="replace")
            )
        except OSError as exc:
            # A completed VASP calculation is not evidence that its benchmark
            # performance record exists.  Never publish fabricated zero-valued
            # timing data for an unreadable required artifact.
            self._record(
                VaspBenchmarkResult(
                    benchmark_id=str(record.metadata["benchmark_id"]),
                    attempt_id=record.attempt_id,
                    outcome=BenchmarkOutcome.FAILED,
                    provenance=provenance,
                    failure_kind="missing_performance_evidence",
                    failure_reason=f"Could not read completed OUTCAR: {exc}",
                )
            )
            return
        if (
            not performance.loop_times
            or performance.total_ranks is None
            or performance.irreducible_kpoints is None
            or performance.electrons is None
        ):
            self._record(
                VaspBenchmarkResult(
                    benchmark_id=str(record.metadata["benchmark_id"]),
                    attempt_id=record.attempt_id,
                    outcome=BenchmarkOutcome.FAILED,
                    provenance=provenance,
                    failure_kind="missing_performance_evidence",
                    failure_reason="Completed OUTCAR contains no benchmark loop timing evidence",
                )
            )
            return
        self._record(
            VaspBenchmarkResult(
                benchmark_id=str(record.metadata["benchmark_id"]),
                attempt_id=record.attempt_id,
                outcome=BenchmarkOutcome.COMPLETED,
                provenance=provenance,
                performance=performance,
            )
        )

    def record_failure(self, record: DftExecutionRecord, failure: Any) -> None:
        outcome = (
            BenchmarkOutcome.OUT_OF_MEMORY
            if getattr(failure, "kind", None) == "out_of_memory"
            else BenchmarkOutcome.FAILED
        )
        self._record(
            VaspBenchmarkResult(
                benchmark_id=str(record.metadata["benchmark_id"]),
                attempt_id=record.attempt_id,
                outcome=outcome,
                provenance=_provenance_from_record(record),
                failure_kind=getattr(failure, "kind", None),
                failure_reason=getattr(failure, "reason", None),
            )
        )

    def _record(self, result: VaspBenchmarkResult) -> None:
        self.results[result.benchmark_id] = result
        append_event = getattr(self.state_store, "append_event", None)
        if append_event is not None:
            append_event(
                f"vasp-benchmark:{result.benchmark_id}:{result.attempt_id}",
                "vasp_benchmark",
                result.benchmark_id,
                "result_recorded",
                result.to_dict(),
            )


def _provenance_from_record(record: DftExecutionRecord) -> VaspBenchmarkProvenance:
    metadata = record.metadata
    resources = record.resources
    if resources is None:
        raise ValueError("benchmark execution record has no JobResources")
    return VaspBenchmarkProvenance(
        compatibility_key=str(metadata["compatibility_key"]),
        structure_id=record.inputs.calculation.structure_id,
        calculation_id=record.inputs.calculation.calculation_id,
        site=metadata.get("site"),
        hardware=metadata.get("hardware"),
        executable_identity=metadata.get("executable_identity"),
        resources=resources,
    )


class VaspBenchmarkRunner:
    """Prepare and reconcile benchmark cases through canonical DFT seams."""

    def __init__(
        self,
        plan: VaspBenchmarkPlan,
        *,
        backend: DftBackend,
        scheduler: Scheduler,
        state_store: Any,
        max_concurrent: int = 1,
    ) -> None:
        self.plan = plan
        self.backend = backend
        self.scheduler = scheduler
        self.state_store = state_store
        self.max_concurrent = max_concurrent
        self._executions: tuple[VaspBenchmarkExecution, ...] | None = None

    def prepare(self) -> tuple[VaspBenchmarkExecution, ...]:
        """Prepare each case with VaspBackend and render only its runner body."""

        from nepflow.stages.dft.reconciliation import DftExecutionRecord

        self.plan.root_directory.mkdir(parents=True, exist_ok=True)
        render_runner = getattr(self.backend, "render_runner_script", None)
        if render_runner is None:
            raise TypeError("VASP benchmark backend must render its runner body")
        runner_path = self.runner_path
        runner_path.write_text(
            "#!/usr/bin/env bash\nset -e\n" + render_runner(),
            encoding="utf-8",
        )
        executions = [self._prepare_case(case, DftExecutionRecord) for case in self.plan.cases]
        self._executions = tuple(executions)
        return self._executions

    def _prepare_case(self, case: VaspBenchmarkCase, record_type: Any) -> VaspBenchmarkExecution:
        case.working_directory.mkdir(parents=True, exist_ok=True)
        upsert_structure = getattr(self.state_store, "upsert_structure", None)
        if upsert_structure is not None:
            upsert_structure(case.target.structure, metadata={"benchmark": True})
        request = DftInputRequest(
            structure=case.target.structure,
            source_structure=case.target.source_structure,
            source_structure_index=case.target.source_structure_index,
            working_directory=case.working_directory,
        )
        inputs = self.backend.prepare_inputs(request)
        incar_path = case.working_directory / "INCAR"
        incar_path.write_text(
            set_incar_parameters(
                incar_path.read_text(encoding="utf-8"),
                case.parameters,
            ),
            encoding="utf-8",
        )
        provenance = VaspBenchmarkProvenance(
            compatibility_key=case.target.compatibility_key,
            structure_id=inputs.calculation.structure_id,
            calculation_id=inputs.calculation.calculation_id,
            site=self.plan.site,
            hardware=self.plan.hardware,
            executable_identity=self.plan.executable_identity,
            resources=case.resource.resources,
        )
        metadata = {
            "execution_kind": "vasp_benchmark",
            "benchmark_id": case.benchmark_id,
            "compatibility_key": case.target.compatibility_key,
            "scientific_identity": inputs.calculation.to_dict(),
            "site": self.plan.site,
            "hardware": self.plan.hardware,
            "executable_identity": self.plan.executable_identity,
            "benchmark_parameters": case.parameters,
        }
        existing = self._existing_attempt(inputs.calculation.calculation_id, case.benchmark_id)
        status = "pending" if existing is None else str(existing.get("status", "pending"))
        if status == "prepared":
            status = "pending"
        record = record_type(
            inputs=inputs,
            attempt_id=(
                str(existing["attempt_id"])
                if existing is not None
                else f"{case.benchmark_id}:attempt:1"
            ),
            status=status,
            job_id=None if existing is None else existing.get("job_id"),
            job_name=case.job_name,
            resources=case.resource.resources,
            metadata=metadata,
        )
        return VaspBenchmarkExecution(case, inputs, record, provenance)

    @property
    def runner_path(self) -> Path:
        return self.plan.root_directory / "run_vasp_benchmark.sh"

    def _existing_attempt(self, calculation_id: str, benchmark_id: str) -> Mapping[str, Any] | None:
        list_attempts = getattr(self.state_store, "list_dft_attempts", None)
        if list_attempts is None:
            return None
        attempts = list_attempts(calculation_id)
        for attempt in reversed(attempts):
            metadata = attempt.get("metadata", {})
            if isinstance(metadata, Mapping) and metadata.get("benchmark_id") == benchmark_id:
                return attempt
        return None

    def reconcile_once(self) -> VaspBenchmarkPass:
        from nepflow.stages.dft.reconciliation import DftReconciliationOrchestrator

        executions = self._executions or self.prepare()
        recorder = _BenchmarkPerformanceRecorder(self.state_store, self.plan)
        self._load_persisted_results(recorder)
        reconciler = DftReconciliationOrchestrator(
            scheduler=self.scheduler,
            backend=self.backend,
            state_store=self.state_store,
            max_concurrent=self.max_concurrent,
            retry_limit=0,
            script_path=self.runner_path,
            performance_recorder=recorder,
            job_name_prefix="nfbench_",
            persist_result_artifact=False,
        )
        execution = reconciler.reconcile_once(tuple(item.record for item in executions))
        by_id = {item.case.benchmark_id: item for item in executions}
        results: list[VaspBenchmarkResult] = []
        for record in execution.records:
            case_id = str(record.metadata["benchmark_id"])
            result = recorder.results.get(case_id)
            if result is None:
                existing = self._load_case_result(case_id)
                if existing is not None:
                    result = existing
            if result is None:
                outcome = {
                    "pending": BenchmarkOutcome.PENDING,
                    "submitted": BenchmarkOutcome.SUBMITTED,
                    "running": BenchmarkOutcome.RUNNING,
                }.get(record.status, BenchmarkOutcome.FAILED)
                result = VaspBenchmarkResult(
                    benchmark_id=case_id,
                    attempt_id=record.attempt_id,
                    outcome=outcome,
                    provenance=by_id[case_id].provenance,
                )
            results.append(result)
        return VaspBenchmarkPass(execution, tuple(results))

    def _load_case_result(self, benchmark_id: str) -> VaspBenchmarkResult | None:
        events = getattr(self.state_store, "list_events", None)
        if events is None:
            return None
        rows = events(entity_type="vasp_benchmark", entity_id=benchmark_id)
        if not rows:
            return None
        return _result_from_dict(rows[-1].get("payload", {}))

    def _load_persisted_results(self, recorder: _BenchmarkPerformanceRecorder) -> None:
        for case in self.plan.cases:
            result = self._load_case_result(case.benchmark_id)
            if result is not None:
                recorder.results[case.benchmark_id] = result


def _result_from_dict(payload: Mapping[str, Any]) -> VaspBenchmarkResult:
    provenance_data = payload["provenance"]
    resources = JobResources(**dict(provenance_data["resources"]))
    performance_data = payload.get("performance")
    performance = None
    if performance_data is not None:
        if not isinstance(performance_data, Mapping):
            raise StateError("VASP benchmark performance evidence must be an object")
        required = (
            "total_ranks",
            "mpi_ranks",
            "loop_times",
            "irreducible_kpoints",
            "electrons",
        )
        missing = [key for key in required if key not in performance_data]
        if missing:
            raise StateError(
                "VASP benchmark performance evidence is missing: " + ", ".join(missing)
            )
        if any(
            performance_data[key] is None
            for key in ("total_ranks", "loop_times", "irreducible_kpoints", "electrons")
        ):
            raise StateError("VASP benchmark performance evidence is incomplete")
        try:
            performance = VaspPerformanceEvidence(
                total_ranks=int(performance_data["total_ranks"]),
                mpi_ranks=int(performance_data["mpi_ranks"]),
                loop_times=tuple(float(value) for value in performance_data["loop_times"]),
                irreducible_kpoints=int(performance_data["irreducible_kpoints"]),
                electrons=float(performance_data["electrons"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise StateError("Malformed VASP benchmark performance evidence") from exc
    return VaspBenchmarkResult(
        benchmark_id=str(payload["benchmark_id"]),
        attempt_id=str(payload["attempt_id"]),
        outcome=BenchmarkOutcome(str(payload["outcome"])),
        provenance=VaspBenchmarkProvenance(
            compatibility_key=str(provenance_data["compatibility_key"]),
            structure_id=str(provenance_data["structure_id"]),
            calculation_id=str(provenance_data["calculation_id"]),
            site=provenance_data.get("site"),
            hardware=provenance_data.get("hardware"),
            executable_identity=provenance_data.get("executable_identity"),
            resources=resources,
        ),
        performance=performance,
        failure_kind=payload.get("failure_kind"),
        failure_reason=payload.get("failure_reason"),
    )


__all__ = [
    "BenchmarkOutcome",
    "VaspBenchmarkCase",
    "VaspBenchmarkExecution",
    "VaspBenchmarkPass",
    "VaspBenchmarkPlan",
    "VaspBenchmarkProvenance",
    "VaspBenchmarkResource",
    "VaspBenchmarkResult",
    "VaspBenchmarkRunner",
    "VaspBenchmarkTarget",
    "build_benchmark_resources",
    "build_vasp_benchmark_plan",
    "build_vasp_benchmark_plan_from_config",
    "compatible_benchmark_observations",
]
