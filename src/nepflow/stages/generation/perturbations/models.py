"""Typed configuration and worker-task records for perturbation generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from nepflow.config.models import DEFAULT_SOURCE_SCOPE, SourceScope
from nepflow.domain.structures import GeneratedStructureRecord

PERTURBATION_FAMILY_SOURCE_FIELDS = {
    "volume_profile": "volume_sources",
    "elastic_stress": "elastic_sources",
    "rattled": "rattle_sources",
    "liquid": "liquid_sources",
    "vacancy": "vacancy_sources",
    "interstitial": "interstitial_sources",
    "gas_interstitial": "gas_interstitial_sources",
    "vacancy_interstitial": "vacancy_interstitial_sources",
    "gas_in_vacancy": "gas_in_vacancy_sources",
}


def derive_child_seed(
    base_structure_id: str,
    root_seed: int,
    family: str,
    slot: int | str = 0,
) -> int:
    """Derive a stable 32-bit seed for one family/output slot."""

    payload = "\x00".join(
        (str(base_structure_id), str(int(root_seed)), str(family), str(slot))
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)


@dataclass(frozen=True, slots=True)
class PerturbationCounts:
    """Per-base candidate counts supplied by the generation configuration."""

    n_rattled: int = 10
    n_liquid_configurations: int = 0
    n_liquid_snapshots: int = 0
    n_vacancies: int = 10
    n_interstitials: int = 10
    n_gas_interstitials: int = 0
    n_vacancy_interstitial: int = 0
    n_gas_in_vacancy: int = 0


@dataclass(frozen=True, slots=True)
class PerturbationSettings:
    """Scientific settings shared by one worker task.

    Distances and rattle amplitudes are in Angstrom, volume scales and strains
    are dimensionless, temperature is kelvin, timestep is femtoseconds, and
    ``target_n_atoms``/step/count fields are integers.  ``random_seed`` is
    persisted into task provenance so worker output is reproducible.
    """

    rattle_std: float = 0.03
    rattle_std_min: float | None = None
    rattle_std_max: float | None = None
    rattle_d_min: float = 1.5
    vacancy_range: tuple[float, float] = (0.0, 0.1)
    interstitial_range: tuple[float, float] = (0.05, 0.1)
    interstitial_d_min: float = 1.65
    volume_scale_range: tuple[float, float] = (0.8, 1.2)
    n_volume_points: int = 11
    target_n_atoms: int = 250
    random_seed: int = 42
    gas_elements: tuple[str, ...] = ()
    gas_interstitial_d_min: float | None = None
    max_gas_occupancy: int = 3
    elastic_stress_enabled: bool = True
    elastic_strain_amplitudes: tuple[float, ...] = (
        -0.02,
        -0.01,
        -0.005,
        0.005,
        0.01,
        0.02,
    )
    liquid_enabled: bool = False
    liquid_temperature_k: float = 3000.0
    liquid_timestep_fs: float = 1.0
    liquid_equilibration_steps: int = 200
    liquid_steps_between_snapshots: int = 100
    liquid_friction: float = 0.02
    volume_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    elastic_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    rattle_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    liquid_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    vacancy_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    interstitial_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    gas_interstitial_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    vacancy_interstitial_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    gas_in_vacancy_sources: SourceScope = DEFAULT_SOURCE_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "vacancy_range", tuple(self.vacancy_range))
        object.__setattr__(self, "interstitial_range", tuple(self.interstitial_range))
        object.__setattr__(self, "volume_scale_range", tuple(self.volume_scale_range))
        object.__setattr__(self, "gas_elements", tuple(self.gas_elements or ()))
        amplitudes = self.elastic_strain_amplitudes
        if amplitudes is None:
            amplitudes = (-0.02, -0.01, -0.005, 0.005, 0.01, 0.02)
        object.__setattr__(self, "elastic_strain_amplitudes", tuple(amplitudes))
        for field_name in PERTURBATION_FAMILY_SOURCE_FIELDS.values():
            scope = getattr(self, field_name)
            if isinstance(scope, str):
                normalized_scope = (scope.strip().lower(),)
            else:
                normalized_scope = tuple(scope)
            object.__setattr__(self, field_name, normalized_scope)
        if self.rattle_std_min is not None and self.rattle_std_max is not None:
            minimum = self.rattle_std_min
            maximum = self.rattle_std_max
        else:
            # Preserve the legacy constructor contract: a range exists only
            # when both endpoints are supplied; otherwise use rattle_std.
            minimum = self.rattle_std
            maximum = self.rattle_std
        object.__setattr__(self, "rattle_std_min", min(minimum, maximum))
        object.__setattr__(self, "rattle_std_max", max(minimum, maximum))
        if self.gas_interstitial_d_min is None:
            object.__setattr__(self, "gas_interstitial_d_min", self.interstitial_d_min)

    def as_engine_kwargs(self) -> dict[str, Any]:
        """Return constructor-compatible values for focused family services."""

        return {
            "rattle_std": self.rattle_std,
            "rattle_std_min": self.rattle_std_min,
            "rattle_std_max": self.rattle_std_max,
            "rattle_d_min": self.rattle_d_min,
            "vacancy_range": self.vacancy_range,
            "interstitial_range": self.interstitial_range,
            "interstitial_d_min": self.interstitial_d_min,
            "volume_scale_range": self.volume_scale_range,
            "n_volume_points": self.n_volume_points,
            "target_n_atoms": self.target_n_atoms,
            "random_seed": self.random_seed,
            "gas_elements": list(self.gas_elements),
            "gas_interstitial_d_min": self.gas_interstitial_d_min,
            "max_gas_occupancy": self.max_gas_occupancy,
            "elastic_stress_enabled": self.elastic_stress_enabled,
            "elastic_strain_amplitudes": list(self.elastic_strain_amplitudes),
            "liquid_enabled": self.liquid_enabled,
            "liquid_temperature_k": self.liquid_temperature_k,
            "liquid_timestep_fs": self.liquid_timestep_fs,
            "liquid_equilibration_steps": self.liquid_equilibration_steps,
            "liquid_steps_between_snapshots": self.liquid_steps_between_snapshots,
            "liquid_friction": self.liquid_friction,
            "volume_sources": list(self.volume_sources),
            "elastic_sources": list(self.elastic_sources),
            "rattle_sources": list(self.rattle_sources),
            "liquid_sources": list(self.liquid_sources),
            "vacancy_sources": list(self.vacancy_sources),
            "interstitial_sources": list(self.interstitial_sources),
            "gas_interstitial_sources": list(self.gas_interstitial_sources),
            "vacancy_interstitial_sources": list(self.vacancy_interstitial_sources),
            "gas_in_vacancy_sources": list(self.gas_in_vacancy_sources),
        }

    def sources_for_family(self, family: str) -> SourceScope:
        """Return the explicit source scope configured for one family."""

        field_name = PERTURBATION_FAMILY_SOURCE_FIELDS.get(family)
        if field_name is None:
            raise KeyError(f"Unknown perturbation family: {family}")
        return getattr(self, field_name)


@dataclass(frozen=True, slots=True)
class PerturbationTask:
    """A complete deterministic unit of worker execution.

    ``Atoms`` is deliberately retained as the scientific input rather than a
    path or an implicit positional tuple.  ASE structures are pickleable, so
    this record is serializable for ``ProcessPoolExecutor`` while preserving
    the exact base identity and effective seed.
    """

    base: Any
    base_structure_id: str
    settings: PerturbationSettings
    counts: PerturbationCounts
    seed: int


@dataclass(frozen=True, slots=True)
class PerturbationRejection:
    """Serializable evidence for one generated candidate that was rejected."""

    parent_structure_id: str
    family: str
    operation_id: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    slot: int | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", dict(self.evidence))

    @property
    def base_structure_id(self) -> str:
        """Return the parent identity under the generation terminology."""

        return self.parent_structure_id

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-shaped rejection evidence for reports and persistence."""

        return {
            "parent_structure_id": self.parent_structure_id,
            "family": self.family,
            "operation_id": self.operation_id,
            "slot": self.slot,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class PerturbationTaskResult:
    """Typed result for one worker task."""

    task: PerturbationTask
    candidates: tuple[Any, ...]
    provenance_records: tuple[GeneratedStructureRecord, ...] = ()
    rejected_attempts: tuple[PerturbationRejection, ...] = ()

    @property
    def rejections(self) -> tuple[PerturbationRejection, ...]:
        """Compatibility alias for callers that use the shorter term."""

        return self.rejected_attempts


__all__ = [
    "PerturbationCounts",
    "PerturbationSettings",
    "PerturbationTask",
    "PerturbationTaskResult",
    "PerturbationRejection",
    "PERTURBATION_FAMILY_SOURCE_FIELDS",
    "derive_child_seed",
]
