"""Canonical validation identity, case, and backend preparation boundary."""

from .preparation import (
    calculate_cell_replicates_for_cutoff,
    cell_perpendicular_heights_angstrom,
    prepare_validation_cases,
)
from .protocols import (
    VALIDATION_CASE_SCHEMA,
    VALIDATION_PREPARATION_SCHEMA,
    ValidationCase,
    ValidationCaseSpec,
    ValidationPreparation,
    ValidationReference,
)
from .resolution import ResolvedModelDataset, resolve_model_dataset
from .stage import ValidationStage, ValidationStageResult
from .metrics import (
    PairedValidationCase,
    ValidationMetrics,
    calculate_metrics,
    calculate_validation_metrics,
    pair_prediction,
    pair_prediction_to_reference,
)
from .reconciliation import (
    ValidationAttempt,
    ValidationCampaign,
    ValidationCaseExecution,
    ValidationExecutionRecord,
    ValidationReconciler,
    ValidationReconciliationOrchestrator,
    ValidationReconciliationResult,
)

__all__ = [
    "VALIDATION_CASE_SCHEMA",
    "VALIDATION_PREPARATION_SCHEMA",
    "ResolvedModelDataset",
    "ValidationCaseSpec",
    "ValidationCase",
    "ValidationPreparation",
    "ValidationReference",
    "ValidationStage",
    "ValidationStageResult",
    "PairedValidationCase",
    "ValidationMetrics",
    "calculate_metrics",
    "calculate_validation_metrics",
    "pair_prediction",
    "pair_prediction_to_reference",
    "ValidationAttempt",
    "ValidationCampaign",
    "ValidationCaseExecution",
    "ValidationExecutionRecord",
    "ValidationReconciler",
    "ValidationReconciliationOrchestrator",
    "ValidationReconciliationResult",
    "calculate_cell_replicates_for_cutoff",
    "cell_perpendicular_heights_angstrom",
    "prepare_validation_cases",
    "resolve_model_dataset",
]
