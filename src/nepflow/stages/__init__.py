"""Canonical workflow stages."""

from .dft import DftStage, DftStageResult
try:
    from .generation import GenerationStage, GenerationResult
except ModuleNotFoundError as exc:
    # Validation and MLIP subpackages are useful in lightweight environments
    # that do not install the optional Materials Project stack.  Keep those
    # canonical boundaries importable without hiding unrelated import errors.
    if exc.name != "pymatgen":
        raise
    GenerationStage = None  # type: ignore[assignment,misc]
    GenerationResult = None  # type: ignore[assignment,misc]
from .training import TrainingStage
from .validation import (
    ValidationCaseSpec,
    ValidationCase,
    ValidationPreparation,
    ValidationReference,
    ValidationStage,
    ValidationStageResult,
    prepare_validation_cases,
    resolve_model_dataset,
)

__all__ = [
    "DftStage",
    "DftStageResult",
    "GenerationStage",
    "GenerationResult",
    "TrainingStage",
    "ValidationCaseSpec",
    "ValidationCase",
    "ValidationPreparation",
    "ValidationReference",
    "ValidationStage",
    "ValidationStageResult",
    "prepare_validation_cases",
    "resolve_model_dataset",
]
