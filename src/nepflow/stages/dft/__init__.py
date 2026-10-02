"""Backend-neutral density-functional-theory workflow stage."""

from .orchestrator import (
    DftCalculationSpec,
    DftPreparationResult,
    PreparedCalculation,
    VaspPreparationOrchestrator,
    calculation_identities_match,
    prepare_calculations,
)
from .stage import DftStage, DftStageResult
from .reconciliation import (
    DftExecutionRecord,
    DftRecoveryDecision,
    DftRecoveryPolicy,
    DftReconciliationOrchestrator,
    DftReconciliationResult,
)
from .reports import (
    DftPerformanceRecord,
    DftPerformanceRecorder,
    VaspBenchmarkSummary,
    benchmark_plot_data,
    summarize_benchmark_results,
)

__all__ = [
    "DftCalculationSpec",
    "DftPreparationResult",
    "DftStage",
    "DftStageResult",
    "DftExecutionRecord",
    "DftRecoveryDecision",
    "DftRecoveryPolicy",
    "DftReconciliationOrchestrator",
    "DftReconciliationResult",
    "DftPerformanceRecord",
    "DftPerformanceRecorder",
    "VaspBenchmarkSummary",
    "benchmark_plot_data",
    "summarize_benchmark_results",
    "PreparedCalculation",
    "VaspPreparationOrchestrator",
    "calculation_identities_match",
    "prepare_calculations",
]
