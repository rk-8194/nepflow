"""Typed DFT execution evidence for reporting and benchmark presentation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ase.io import read as ase_read

from nepflow.dft.backend import DftFailure
from nepflow.dft.vasp.outputs import parse_performance_evidence
from nepflow.dft.vasp.resource_memory import (
    VaspMemoryObservation,
    append_vasp_memory,
    estimate_n_electrons,
    estimate_kpoints_irr,
    read_incar_resource_parameters,
    resolve_memory_path,
)
from nepflow.errors import VaspError

from .reconciliation import DftExecutionRecord


@dataclass(frozen=True, slots=True)
class DftPerformanceRecord:
    """Stable performance evidence associated with one DFT attempt."""

    calculation_id: str
    attempt_id: str
    status: str
    total_ranks: int | None = None
    mpi_ranks: int | None = None
    irreducible_kpoints: int | None = None
    electrons: float | None = None
    average_loop_time: float | None = None
    failure_kind: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation_id": self.calculation_id,
            "attempt_id": self.attempt_id,
            "status": self.status,
            "total_ranks": self.total_ranks,
            "mpi_ranks": self.mpi_ranks,
            "irreducible_kpoints": self.irreducible_kpoints,
            "electrons": self.electrons,
            "average_loop_time": self.average_loop_time,
            "failure_kind": self.failure_kind,
            "failure_reason": self.failure_reason,
        }


class DftPerformanceSink(Protocol):
    def record_completion(
        self,
        record: DftExecutionRecord,
        result: Any,
    ) -> None: ...

    def record_failure(
        self,
        record: DftExecutionRecord,
        failure: DftFailure,
    ) -> None: ...


class DftPerformanceRecorder:
    """Persist backend performance/failure evidence as idempotent events."""

    def __init__(
        self,
        state_store: Any,
        *,
        memory_path: Any | None = None,
        project_root: Any | None = None,
        config: Any | None = None,
    ) -> None:
        self.state_store = state_store
        self.project_root = None if project_root is None else Path(project_root)
        self.memory_path = (
            Path(memory_path)
            if memory_path is not None
            else (resolve_memory_path(self.project_root) if self.project_root is not None else None)
        )
        self.config = config

    def record_completion(self, record: DftExecutionRecord, result: Any) -> None:
        if record.status == "reused":
            return
        outcar = record.inputs.working_directory / "OUTCAR"
        try:
            evidence = parse_performance_evidence(
                outcar.read_text(encoding="utf-8", errors="replace")
            )
        except OSError as exc:
            # Completion of the calculation does not make an unreadable
            # performance artifact valid.  Preserve the DFT result while
            # recording missing optional reporting evidence explicitly.
            self._record(
                record,
                DftPerformanceRecord(
                    calculation_id=record.inputs.calculation.calculation_id,
                    attempt_id=record.attempt_id,
                    status="completed",
                    failure_kind="missing_performance_evidence",
                    failure_reason=f"Could not read completed OUTCAR: {exc}",
                ),
                result,
            )
            return
        if not evidence.loop_times:
            self._record(
                record,
                DftPerformanceRecord(
                    calculation_id=record.inputs.calculation.calculation_id,
                    attempt_id=record.attempt_id,
                    status="completed",
                    failure_kind="missing_performance_evidence",
                    failure_reason="Completed OUTCAR contains no performance loop timing evidence",
                ),
                result,
            )
            return
        performance = DftPerformanceRecord(
            calculation_id=record.inputs.calculation.calculation_id,
            attempt_id=record.attempt_id,
            status="completed",
            total_ranks=evidence.total_ranks,
            mpi_ranks=evidence.mpi_ranks,
            irreducible_kpoints=evidence.irreducible_kpoints,
            electrons=evidence.electrons,
            average_loop_time=evidence.average_loop_time,
        )
        self._append_memory_observation(record, evidence, oom=0)
        self._record(record, performance, result)

    def record_failure(self, record: DftExecutionRecord, failure: DftFailure) -> None:
        if failure.kind == "out_of_memory":
            evidence = self._read_optional_evidence(record)
            self._append_memory_observation(record, evidence, oom=1)
        self._record(
            record,
            DftPerformanceRecord(
                calculation_id=record.inputs.calculation.calculation_id,
                attempt_id=record.attempt_id,
                status="failed",
                failure_kind=failure.kind,
                failure_reason=failure.reason,
            ),
            None,
        )

    @staticmethod
    def _read_optional_evidence(record: DftExecutionRecord):
        try:
            return parse_performance_evidence(
                (record.inputs.working_directory / "OUTCAR").read_text(
                    encoding="utf-8", errors="replace"
                )
            )
        except OSError:
            return None

    def _append_memory_observation(self, record: DftExecutionRecord, evidence: Any, *, oom: int) -> None:
        if self.memory_path is None:
            return
        try:
            atoms = len(ase_read(str(record.inputs.working_directory / "POSCAR")))
            observed_kpoints = None if evidence is None else evidence.irreducible_kpoints
            observed_electrons = None if evidence is None else evidence.electrons
            if observed_kpoints is None:
                if self.project_root is None:
                    return
                observed_kpoints = estimate_kpoints_irr(
                    record.inputs.working_directory / "POSCAR",
                    self.project_root,
                    config=self.config,
                    incar_path=record.inputs.working_directory / "INCAR",
                )
            if observed_electrons is None:
                if self.project_root is None:
                    return
                observed_electrons = estimate_n_electrons(
                    record.inputs.working_directory / "POSCAR", self.project_root
                )
            ncore, kpar = read_incar_resource_parameters(record.inputs.working_directory / "INCAR")
            resources = record.resources
            if resources is None:
                return
            loop_time = 0.0 if oom else (0.0 if evidence is None else evidence.average_loop_time)
            if not loop_time and oom == 0:
                return
            append_vasp_memory(
                self.memory_path,
                VaspMemoryObservation(
                    n_atoms=atoms,
                    n_kpoints_irr=int(observed_kpoints or 0),
                    n_electrons=int(round(float(observed_electrons or 0))),
                    nodes=resources.nodes,
                    total_gpus=resources.total_gpus,
                    ncore=ncore,
                    kpar=kpar,
                    avg_loop_time=float(loop_time),
                    oom=oom,
                ),
            )
        except (OSError, UnicodeError, ValueError, VaspError) as exc:
            # Performance provenance remains in StateStore; a missing optional
            # append must not erase or reclassify a completed DFT result.
            self._record_memory_append_failure(record, exc)

    def _record_memory_append_failure(self, record: DftExecutionRecord, exc: Exception) -> None:
        self.state_store.append_event(
            f"dft-performance-memory:{record.attempt_id}",
            "dft_attempt",
            record.attempt_id,
            "legacy_memory_append_failed",
            {"error": str(exc), "path": None if self.memory_path is None else str(self.memory_path)},
        )

    def _record(
        self,
        record: DftExecutionRecord,
        performance: DftPerformanceRecord,
        result: Any,
    ) -> None:
        payload = performance.to_dict()
        if result is not None and hasattr(result, "artifact"):
            payload["artifact"] = None if result.artifact is None else result.artifact.to_dict()
        self.state_store.append_event(
            f"dft-performance:{record.attempt_id}:{performance.status}",
            "dft_attempt",
            record.attempt_id,
            "performance_recorded",
            payload,
        )


@dataclass(frozen=True, slots=True)
class VaspBenchmarkSummary:
    """Structured benchmark summary used by DFT reports and plot adapters."""

    total: int
    completed: int
    out_of_memory: int
    failed: int
    average_loop_time: float | None


def summarize_benchmark_results(results: Sequence[Any]) -> VaspBenchmarkSummary:
    """Summarize typed benchmark results without reading launcher output."""

    completed = [
        result
        for result in results
        if getattr(getattr(result, "outcome", None), "value", None) == "completed"
        and getattr(result, "performance", None) is not None
    ]
    loop_times = [
        result.performance.average_loop_time
        for result in completed
        if result.performance.average_loop_time > 0
    ]
    return VaspBenchmarkSummary(
        total=len(results),
        completed=len(completed),
        out_of_memory=sum(
            getattr(getattr(result, "outcome", None), "value", None) == "out_of_memory"
            for result in results
        ),
        failed=sum(
            getattr(getattr(result, "outcome", None), "value", None) == "failed"
            for result in results
        ),
        average_loop_time=(sum(loop_times) / len(loop_times) if loop_times else None),
    )


def benchmark_plot_data(results: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    """Return plot-ready structured rows; plotting libraries stay out of DFT state."""

    return tuple(
        {
            "benchmark_id": result.benchmark_id,
            "outcome": result.outcome.value,
            "structure_id": result.provenance.structure_id,
            "compatibility_key": result.provenance.compatibility_key,
            "nodes": result.provenance.resources.nodes,
            "gpus_per_node": result.provenance.resources.gpus_per_node,
            "total_gpus": result.provenance.resources.total_gpus,
            "mpi_ranks": result.provenance.resources.mpi_ranks,
            "average_loop_time": (
                None if result.performance is None else result.performance.average_loop_time
            ),
        }
        for result in results
    )


__all__ = [
    "DftPerformanceRecord",
    "DftPerformanceRecorder",
    "DftPerformanceSink",
    "VaspBenchmarkSummary",
    "benchmark_plot_data",
    "summarize_benchmark_results",
]
