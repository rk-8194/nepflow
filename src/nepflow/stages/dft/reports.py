"""Typed DFT execution evidence for reporting and benchmark presentation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from nepflow.dft.backend import DftFailure
from nepflow.dft.vasp.outputs import parse_performance_evidence

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

    def __init__(self, state_store: Any) -> None:
        self.state_store = state_store

    def record_completion(self, record: DftExecutionRecord, result: Any) -> None:
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
        if (
            not evidence.loop_times
            or evidence.total_ranks is None
            or evidence.irreducible_kpoints is None
            or evidence.electrons is None
        ):
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
        self._record(record, performance, result)

    def record_failure(self, record: DftExecutionRecord, failure: DftFailure) -> None:
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
