"""Canonical NEP training-stage boundaries."""

from .campaign import (
    CampaignReconciliationResult,
    TrainingAttempt,
    TrainingCampaign,
    TrainingCandidate,
)
from .dataset import (
    DatasetBuildReport,
    DatasetBuildResult,
    DatasetSplit,
    build_dataset_metadata,
    build_training_dataset,
    load_materialized_dataset,
    resolve_selected_dft_results,
    write_nep_dataset,
)
from .optimisation import (
    CandidateConfiguration,
    CandidateSweep,
    ControlledCandidateSweep,
    ControlledSweep,
    generate_candidate_matrix,
)
from .reports import TrainingCampaignReport, summarize_campaign
from .stage import TrainingStage

__all__ = [
    "DatasetBuildReport",
    "DatasetBuildResult",
    "DatasetSplit",
    "CampaignReconciliationResult",
    "CandidateConfiguration",
    "CandidateSweep",
    "ControlledCandidateSweep",
    "ControlledSweep",
    "TrainingAttempt",
    "TrainingCandidate",
    "TrainingCampaign",
    "TrainingCampaignReport",
    "TrainingStage",
    "build_dataset_metadata",
    "build_training_dataset",
    "generate_candidate_matrix",
    "load_materialized_dataset",
    "resolve_selected_dft_results",
    "summarize_campaign",
    "write_nep_dataset",
]
