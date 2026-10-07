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
    """Counts and budget information for one structural parent."""

    total_available_afm: int = 0
    retained_afm: int = 0
    budget_truncated: bool = False
    diagnostics: tuple[MagneticGenerationDiagnostic, ...] = ()

    @property
    def omitted_afm(self) -> int:
        return max(0, self.total_available_afm - self.retained_afm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_available_afm": self.total_available_afm,
            "retained_afm": self.retained_afm,
            "omitted_afm": self.omitted_afm,
            "budget_truncated": self.budget_truncated,
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
