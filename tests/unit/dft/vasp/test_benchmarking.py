"""Focused tests for the explicit VASP benchmark service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nepflow.dft.backend import DftCompletionEvidence, DftFailure, DftInputArtifacts, DftInputRequest
from nepflow.dft.vasp.benchmarking import (
    BenchmarkOutcome,
    VaspBenchmarkProvenance,
    VaspBenchmarkResult,
    VaspBenchmarkRunner,
    VaspBenchmarkTarget,
    build_benchmark_resources,
    build_vasp_benchmark_plan,
    compatible_benchmark_observations,
)
from nepflow.dft.vasp.inputs import hash_incar_text, set_incar_parameters
from nepflow.domain.identities import DftCalculationIdentity, StructureIdentity
from nepflow.hpc.jobs import QueueQueryResult, ReconciledJobResult, SchedulerJobState, SlurmJobRecord, SubmissionResult
from nepflow.stages.dft.reports import summarize_benchmark_results


def _target(source: Path, key: str = "material-a") -> VaspBenchmarkTarget:
    return VaspBenchmarkTarget(
        structure=StructureIdentity("structure-a"),
        source_structure=source,
        compatibility_key=key,
    )


def test_plan_and_resource_grid_are_deterministic(tmp_path: Path) -> None:
    resources = build_benchmark_resources(
        cores_per_node=16,
        gpus_per_node=4,
        walltime="00:10:00",
        memory_per_node="80G",
    )
    first = build_vasp_benchmark_plan(
        project_name="p",
        root_directory=tmp_path / "bench",
        targets=[_target(tmp_path / "POSCAR")],
        resources=resources,
    )
    second = build_vasp_benchmark_plan(
        project_name="p",
        root_directory=tmp_path / "bench",
        targets=[_target(tmp_path / "POSCAR")],
        resources=resources,
    )

    assert first.benchmark_id == second.benchmark_id
    assert [case.benchmark_id for case in first.cases] == [case.benchmark_id for case in second.cases]
    assert all(
        case.resource.resources.total_gpus
        == case.resource.resources.nodes * case.resource.resources.gpus_per_node
        for case in first.cases
    )
    assert all("64G" not in str(case.resource.resources.memory_per_node) for case in first.cases)


def test_resource_only_incar_overrides_preserve_scientific_hash() -> None:
    scientific = "ENCUT = 520\nNCORE = 2\nKPAR = 1\n"
    resource_variant = set_incar_parameters(
        scientific,
        {"NCORE": 16, "KPAR": 4},
    )

    assert hash_incar_text(scientific) == hash_incar_text(resource_variant)
    benchmark = set_incar_parameters(
        resource_variant,
        {"NSW": 0, "NELM": 10},
    )
    assert "NCORE = 16" in benchmark
    assert "KPAR = 4" in benchmark


def test_incompatible_observation_cannot_be_reused(tmp_path: Path) -> None:
    resources = build_benchmark_resources(cores_per_node=8, gpus_per_node=2)
    plan = build_vasp_benchmark_plan(
        project_name="p",
        root_directory=tmp_path / "bench",
        targets=[_target(tmp_path / "POSCAR")],
        resources=resources[:1],
        site="site-a",
        hardware="gpu-a",
        executable_identity="vasp_std@1",
    )
    provenance = VaspBenchmarkProvenance(
        compatibility_key="material-a",
        structure_id="structure-a",
        calculation_id="calculation-a",
        site="site-a",
        hardware="gpu-a",
        executable_identity="vasp_std@1",
        resources=resources[0].resources,
    )
    result = VaspBenchmarkResult(
        plan.cases[0].benchmark_id,
        "attempt",
        BenchmarkOutcome.FAILED,
        provenance,
    )

    assert compatible_benchmark_observations((result,), provenance) == ()
    assert not result.can_reuse_for(provenance)


@dataclass
class FakeStateStore:
    attempts: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def upsert_structure(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def list_dft_attempts(self, calculation_id: str) -> list[dict[str, Any]]:
        return [value for value in self.attempts.values() if value["calculation_id"] == calculation_id]

    def save_execution(self, record, *, artifact=None, reason=None) -> None:
        del artifact, reason
        self.attempts[record.attempt_id] = {
            "attempt_id": record.attempt_id,
            "calculation_id": record.inputs.calculation.calculation_id,
            "status": record.status,
            "job_id": record.job_id,
            "metadata": dict(record.metadata),
        }

    def append_event(self, event_id, entity_type, entity_id, event_type, payload, **_kwargs):
        self.events.append({
            "event_id": event_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "event_type": event_type,
            "payload": payload,
        })

    def list_events(self, *, entity_type=None, entity_id=None):
        return [
            event
            for event in self.events
            if event["entity_type"] == entity_type and event["entity_id"] == entity_id
        ]


class FakeBackend:
    def prepare_inputs(self, request: DftInputRequest) -> DftInputArtifacts:
        request.working_directory.mkdir(parents=True, exist_ok=True)
        (request.working_directory / "INCAR").write_text("ENCUT = 520\n")
        calculation = DftCalculationIdentity(
            structure_id=request.structure.structure_id,
            incar_hash="incar",
            potcar_hash="potcar",
        )
        return DftInputArtifacts(calculation, request.working_directory, ())

    def render_runner_script(self) -> str:
        return "echo benchmark\n"

    def parse_completion(self, inputs, process=None):
        del inputs, process
        return DftCompletionEvidence(False)

    def parse_result(self, inputs):
        raise AssertionError("not used in submission test")

    def classify_failure(self, evidence):
        del evidence
        return DftFailure("out_of_memory", True, "oom")


class FakeScheduler:
    def __init__(self) -> None:
        self.submissions: list[Path] = []
        self.state = SchedulerJobState.PENDING

    def list_active_jobs(self, **_kwargs):
        return QueueQueryResult(())

    def submit_script(self, script_path, **_kwargs):
        self.submissions.append(Path(script_path))
        return SubmissionResult(f"job-{len(self.submissions)}", "", "", ("sbatch", str(script_path)))

    def reconcile(self, job_id: str, **_kwargs) -> ReconciledJobResult:
        return ReconciledJobResult(
            job_id=job_id,
            job=SlurmJobRecord(job_id, "nfbench", self.state),
            source="fake",
        )


def test_restart_does_not_resubmit_and_oom_is_not_success(tmp_path: Path) -> None:
    resources = build_benchmark_resources(cores_per_node=8, gpus_per_node=1)[:1]
    plan = build_vasp_benchmark_plan(
        project_name="p",
        root_directory=tmp_path / "bench",
        targets=[_target(tmp_path / "POSCAR")],
        resources=resources,
    )
    store = FakeStateStore()
    scheduler = FakeScheduler()
    runner = VaspBenchmarkRunner(
        plan,
        backend=FakeBackend(),
        scheduler=scheduler,
        state_store=store,
    )
    first = runner.reconcile_once()
    assert scheduler.submissions == [plan.root_directory / "run_vasp_benchmark.sh"]
    assert first.results[0].outcome is BenchmarkOutcome.SUBMITTED

    second_runner = VaspBenchmarkRunner(
        plan,
        backend=FakeBackend(),
        scheduler=scheduler,
        state_store=store,
    )
    second = second_runner.reconcile_once()
    assert len(scheduler.submissions) == 1
    assert second.results[0].outcome is BenchmarkOutcome.SUBMITTED

    scheduler.state = SchedulerJobState.OOM
    third = second_runner.reconcile_once()
    assert third.results[0].outcome is BenchmarkOutcome.OUT_OF_MEMORY
    assert not third.results[0].successful
    assert summarize_benchmark_results(third.results).completed == 0
