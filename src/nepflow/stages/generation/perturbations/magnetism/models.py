"""Generation-local records for magnetic expansion and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MagneticGenerationDiagnostic:
    """An explicit, reproducible inclusion or exclusion diagnostic."""

    code: str
    message: str
    ordering: str | None = None
    moment_set: str | None = None
    propagation_vector: tuple[float, float, float] | None = None
    parent_structure_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "ordering": self.ordering,
            "moment_set": self.moment_set,
            "propagation_vector": (
                None if self.propagation_vector is None else list(self.propagation_vector)
            ),
            "parent_structure_id": self.parent_structure_id,
        }


@dataclass(frozen=True, slots=True)
class MagneticGenerationSummary:
    """Counts and budget information for one parent or expansion stream."""

    total_available_afm: int = 0
    retained_afm: int = 0
    budget_truncated: bool = False
    diagnostics: tuple[MagneticGenerationDiagnostic, ...] = ()
    structural_parents_examined: int = 0
    selected_structural_parents: int = 0
    defect_budget_excluded_parents: int = 0
    eligible_structural_parents: int = 0
    selected_with_configured_magnetic_sites: int = 0
    selected_without_configured_magnetic_sites: int = 0
    expanded_structural_parents: int = 0
    zero_output_failures: int = 0
    configured_magnetic_site_count: int = 0
    emitted_non_magnetic: int = 0
    emitted_ferromagnetic: int = 0
    emitted_antiferromagnetic: int = 0
    total_magnetic_candidates: int = 0
    final_variant_budget_truncated: bool = False
    final_variant_budget_truncations: int = 0
    zero_variant_eligible_parents: int = 0

    @property
    def afm_budget_truncated(self) -> bool:
        """Whether the AFM-only enumeration budget truncated any states."""

        return self.budget_truncated

    @property
    def omitted_afm(self) -> int:
        return max(0, self.total_available_afm - self.retained_afm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_parents_examined": self.structural_parents_examined,
            "selected_structural_parents": self.selected_structural_parents,
            "defect_budget_excluded_parents": self.defect_budget_excluded_parents,
            "eligible_structural_parents": self.eligible_structural_parents,
            "selected_with_configured_magnetic_sites": (
                self.selected_with_configured_magnetic_sites
            ),
            "selected_without_configured_magnetic_sites": (
                self.selected_without_configured_magnetic_sites
            ),
            "expanded_structural_parents": self.expanded_structural_parents,
            "zero_output_failures": self.zero_output_failures,
            "configured_magnetic_site_count": self.configured_magnetic_site_count,
            "emitted_non_magnetic": self.emitted_non_magnetic,
            "emitted_ferromagnetic": self.emitted_ferromagnetic,
            "emitted_antiferromagnetic": self.emitted_antiferromagnetic,
            "total_magnetic_candidates": self.total_magnetic_candidates,
            "total_available_afm": self.total_available_afm,
            "retained_afm": self.retained_afm,
            "omitted_afm": self.omitted_afm,
            "budget_truncated": self.budget_truncated,
            "afm_budget_truncated": self.afm_budget_truncated,
            "final_variant_budget_truncated": self.final_variant_budget_truncated,
            "final_variant_budget_truncations": self.final_variant_budget_truncations,
            "zero_variant_eligible_parents": self.zero_variant_eligible_parents,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class MagneticGenerationResult:
    """Expanded candidates and diagnostics for one structural parent."""

    candidates: tuple[Any, ...]
    summary: MagneticGenerationSummary = field(default_factory=MagneticGenerationSummary)

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)
