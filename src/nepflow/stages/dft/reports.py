"""Typed DFT execution evidence for reporting and benchmark presentation."""

from __future__ import annotations

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
    total_ranks: int = 0
    mpi_ranks: int = 0
    irreducible_kpoints: int = 0
    electrons: float = 0.0
    average_loop_time: float = 0.0
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
        except OSError:
            evidence = None
        performance = DftPerformanceRecord(
            calculation_id=record.inputs.calculation.calculation_id,
            attempt_id=record.attempt_id,
            status="completed",
            total_ranks=0 if evidence is None else evidence.total_ranks,
            mpi_ranks=0 if evidence is None else evidence.mpi_ranks,
            irreducible_kpoints=(
                0 if evidence is None else evidence.irreducible_kpoints
            ),
            electrons=0.0 if evidence is None else evidence.electrons,
            average_loop_time=(
                0.0 if evidence is None else evidence.average_loop_time
            ),
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
            payload["artifact"] = (
                None if result.artifact is None else result.artifact.to_dict()
            )
        self.state_store.append_event(
            f"dft-performance:{record.attempt_id}:{performance.status}",
            "dft_attempt",
            record.attempt_id,
            "performance_recorded",
            payload,
        )


__all__ = ["DftPerformanceRecord", "DftPerformanceRecorder", "DftPerformanceSink"]
