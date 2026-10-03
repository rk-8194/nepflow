"""Typed configuration and worker-task records for perturbation generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    """Scientific settings shared by one worker task."""

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "vacancy_range", tuple(self.vacancy_range))
        object.__setattr__(self, "interstitial_range", tuple(self.interstitial_range))
        object.__setattr__(self, "volume_scale_range", tuple(self.volume_scale_range))
        object.__setattr__(self, "gas_elements", tuple(self.gas_elements or ()))
        amplitudes = self.elastic_strain_amplitudes
        if amplitudes is None:
            amplitudes = (-0.02, -0.01, -0.005, 0.005, 0.01, 0.02)
        object.__setattr__(self, "elastic_strain_amplitudes", tuple(amplitudes))
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
        }


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
class PerturbationTaskResult:
    """Typed result for one worker task."""

    task: PerturbationTask
    candidates: tuple[Any, ...]
    provenance_records: tuple[Any, ...] = ()


__all__ = [
    "PerturbationCounts",
    "PerturbationSettings",
    "PerturbationTask",
    "PerturbationTaskResult",
]
