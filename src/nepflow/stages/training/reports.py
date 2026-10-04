"""Presentation helpers for persisted training-campaign state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .campaign import CampaignReconciliationResult


@dataclass(frozen=True, slots=True)
class TrainingCampaignReport:
    campaign_id: str
    status: str
    counts: Mapping[str, int]
    completed_model_run_ids: tuple[str, ...]
    failed_model_run_ids: tuple[str, ...]
    promotion_decision: Mapping[str, Any] | None

    @classmethod
    def from_result(
        cls,
        result: CampaignReconciliationResult,
    ) -> "TrainingCampaignReport":
        return cls(
            campaign_id=result.campaign_id,
            status=result.status,
            counts=dict(result.counts),
            completed_model_run_ids=tuple(
                candidate.model_run_id
                for candidate in result.candidates
                if candidate.status == "completed"
            ),
            failed_model_run_ids=tuple(
                candidate.model_run_id
                for candidate in result.candidates
                if candidate.status == "failed"
            ),
            promotion_decision=(
                None if result.promotion_decision is None else dict(result.promotion_decision)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "status": self.status,
            "counts": dict(self.counts),
            "completed_model_run_ids": list(self.completed_model_run_ids),
            "failed_model_run_ids": list(self.failed_model_run_ids),
            "promotion_decision": self.promotion_decision,
        }


def summarize_campaign(result: CampaignReconciliationResult) -> TrainingCampaignReport:
    """Build a presentation-only report without changing campaign state."""

    return TrainingCampaignReport.from_result(result)


__all__ = ["TrainingCampaignReport", "summarize_campaign"]
