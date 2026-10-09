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
    "substitution": "substitution_sources",
    "antisite": "antisite_sources",
    "vacancy_interstitial": "vacancy_interstitial_sources",
    "gas_in_vacancy": "gas_in_vacancy_sources",
    "surface": "surface_sources",
    "grain_boundary": "grain_boundary_sources",
}

# There is intentionally one liquid implementation at present.  Keep its
# method/fidelity labels in the shared model layer so generation, validation,
# and downstream provenance consumers agree on stable values.
LIQUID_METHOD = "ase_langevin_lj"
LIQUID_FIDELITY = "geometry_disorder_only_not_material_specific"


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
    """Per-base candidate counts supplied by the generation configuration.

    ``n_surfaces`` remains for compatibility with the typed coordinator
    interface and old callers. The configured process path is
    orientation/termination driven and does not use this legacy global count
    as a cap.
    """

    n_rattled: int = 10
    n_liquid_configurations: int = 0
    n_liquid_snapshots: int = 0
    n_vacancies: int = 10
    n_interstitials: int = 10
    n_gas_interstitials: int = 0
    n_substitutions: int = 0
    n_antisites: int = 0
    n_vacancy_interstitial: int = 0
    n_gas_in_vacancy: int = 0
    n_surfaces: int = 0
    n_grain_boundaries: int = 0


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
    defect_defect_d_min: float = 0.0
    periodic_image_d_min: float = 0.0
    interstitial_max_attempts: int = 500
    volume_scale_range: tuple[float, float] = (0.8, 1.2)
    n_volume_points: int = 11
    target_n_atoms: int = 250
    random_seed: int = 42
    gas_elements: tuple[str, ...] = ()
    gas_interstitial_d_min: float | None = None
    max_gas_occupancy: int = 3
    vacancy_species: tuple[str, ...] = ()
    substitution_pairs: tuple[tuple[str, str], ...] = ()
    antisite_pairs: tuple[tuple[str, str], ...] = ()
    substitution_range: tuple[float, float] = (0.0, 0.1)
    antisite_range: tuple[float, float] = (0.0, 0.1)
    interstitial_sites: tuple[Any, ...] = ()
    crystallographic_interstitial_sites: tuple[Any, ...] = ()
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
    substitution_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    antisite_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    vacancy_interstitial_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    gas_in_vacancy_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    surface_enabled: bool = False
    surface_miller_indices: tuple[tuple[int, int, int], ...] = ((1, 0, 0),)
    surface_layers: int = 3
    surface_thickness: float | None = None
    # Minimum physical vacuum thickness in Angstrom; never an hkl-plane count.
    surface_vacuum: float = 10.0
    surface_min_half_depth: float = 6.0
    surface_bulk_environment_radius: float = 5.0
    surface_min_bulk_core_atoms: int = 1
    surface_bulk_environment_distance_tolerance: float = 0.05
    surface_termination_policy: str = "all"
    surface_max_terminations: int = 0
    surface_target_n_atoms: int | None = None
    surface_target_tolerance: float = 0.20
    surface_max_n_atoms: int = 512
    surface_in_plane_repeat: tuple[int, int] = (1, 1)
    surface_max_in_plane_repeat: tuple[int, int] = (4, 4)
    surface_max_normal_repeat: int = 16
    surface_min_in_plane_dimensions: tuple[float, float] = (0.0, 0.0)
    surface_symmetric: bool = False
    surface_sources: SourceScope = DEFAULT_SOURCE_SCOPE
    grain_boundary_enabled: bool = False
    grain_boundary_rotation_axis: tuple[int, int, int] = (0, 0, 1)
    grain_boundary_misorientation_angle: float = 36.86989764584402
    grain_boundary_sigma: int = 5
    grain_boundary_plane: tuple[int, int, int] = (2, 1, 0)
    grain_boundary_expand_times: int = 2
    grain_boundary_min_thickness: float = 0.0
    grain_boundary_overlap_tolerance: float = 0.7
    grain_boundary_sources: SourceScope = DEFAULT_SOURCE_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "vacancy_range", tuple(self.vacancy_range))
        object.__setattr__(self, "interstitial_range", tuple(self.interstitial_range))
        object.__setattr__(self, "substitution_range", tuple(self.substitution_range))
        object.__setattr__(self, "antisite_range", tuple(self.antisite_range))
        object.__setattr__(self, "volume_scale_range", tuple(self.volume_scale_range))
        object.__setattr__(
            self,
            "surface_miller_indices",
            tuple(tuple(int(value) for value in index) for index in self.surface_miller_indices),
        )
        object.__setattr__(self, "surface_in_plane_repeat", tuple(self.surface_in_plane_repeat))
        object.__setattr__(
            self, "surface_max_in_plane_repeat", tuple(self.surface_max_in_plane_repeat)
        )
        object.__setattr__(
            self, "grain_boundary_rotation_axis", tuple(self.grain_boundary_rotation_axis)
        )
        object.__setattr__(self, "grain_boundary_plane", tuple(self.grain_boundary_plane))
        object.__setattr__(
            self,
            "surface_min_in_plane_dimensions",
            tuple(self.surface_min_in_plane_dimensions),
        )
        object.__setattr__(self, "gas_elements", tuple(self.gas_elements or ()))
        object.__setattr__(self, "vacancy_species", tuple(self.vacancy_species or ()))
        object.__setattr__(self, "substitution_pairs", _normalise_pairs(self.substitution_pairs))
        object.__setattr__(self, "antisite_pairs", _normalise_pairs(self.antisite_pairs))
        configured_sites = self.crystallographic_interstitial_sites or self.interstitial_sites
        object.__setattr__(self, "interstitial_sites", tuple(configured_sites or ()))
        object.__setattr__(
            self,
            "crystallographic_interstitial_sites",
            tuple(configured_sites or ()),
        )
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
            "defect_defect_d_min": self.defect_defect_d_min,
            "periodic_image_d_min": self.periodic_image_d_min,
            "interstitial_max_attempts": self.interstitial_max_attempts,
            "volume_scale_range": self.volume_scale_range,
            "n_volume_points": self.n_volume_points,
            "target_n_atoms": self.target_n_atoms,
            "random_seed": self.random_seed,
            "gas_elements": list(self.gas_elements),
            "gas_interstitial_d_min": self.gas_interstitial_d_min,
            "max_gas_occupancy": self.max_gas_occupancy,
            "vacancy_species": list(self.vacancy_species),
            "substitution_pairs": [list(pair) for pair in self.substitution_pairs],
            "antisite_pairs": [list(pair) for pair in self.antisite_pairs],
            "substitution_range": self.substitution_range,
            "antisite_range": self.antisite_range,
            "interstitial_sites": list(self.interstitial_sites),
            "crystallographic_interstitial_sites": list(self.crystallographic_interstitial_sites),
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
            "substitution_sources": list(self.substitution_sources),
            "antisite_sources": list(self.antisite_sources),
            "vacancy_interstitial_sources": list(self.vacancy_interstitial_sources),
            "gas_in_vacancy_sources": list(self.gas_in_vacancy_sources),
            "surface_enabled": self.surface_enabled,
            "surface_miller_indices": [list(index) for index in self.surface_miller_indices],
            "surface_layers": self.surface_layers,
            "surface_thickness": self.surface_thickness,
            "surface_vacuum": self.surface_vacuum,
            "surface_min_half_depth": self.surface_min_half_depth,
            "surface_bulk_environment_radius": self.surface_bulk_environment_radius,
            "surface_min_bulk_core_atoms": self.surface_min_bulk_core_atoms,
            "surface_bulk_environment_distance_tolerance": (
                self.surface_bulk_environment_distance_tolerance
            ),
            "surface_termination_policy": self.surface_termination_policy,
            "surface_max_terminations": self.surface_max_terminations,
            "surface_target_n_atoms": self.surface_target_n_atoms,
            "surface_target_tolerance": self.surface_target_tolerance,
            "surface_max_n_atoms": self.surface_max_n_atoms,
            "surface_in_plane_repeat": self.surface_in_plane_repeat,
            "surface_max_in_plane_repeat": self.surface_max_in_plane_repeat,
            "surface_max_normal_repeat": self.surface_max_normal_repeat,
            "surface_min_in_plane_dimensions": self.surface_min_in_plane_dimensions,
            "surface_symmetric": self.surface_symmetric,
            "surface_sources": list(self.surface_sources),
            "grain_boundary_enabled": self.grain_boundary_enabled,
            "grain_boundary_rotation_axis": self.grain_boundary_rotation_axis,
            "grain_boundary_misorientation_angle": self.grain_boundary_misorientation_angle,
            "grain_boundary_sigma": self.grain_boundary_sigma,
            "grain_boundary_plane": self.grain_boundary_plane,
            "grain_boundary_expand_times": self.grain_boundary_expand_times,
            "grain_boundary_min_thickness": self.grain_boundary_min_thickness,
            "grain_boundary_overlap_tolerance": self.grain_boundary_overlap_tolerance,
            "grain_boundary_sources": list(self.grain_boundary_sources),
        }

    def sources_for_family(self, family: str) -> SourceScope:
        """Return the explicit source scope configured for one family."""

        field_name = PERTURBATION_FAMILY_SOURCE_FIELDS.get(family)
        if field_name is None:
            raise KeyError(f"Unknown perturbation family: {family}")
        return getattr(self, field_name)


def _normalise_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    """Normalize configured ``(source, target)`` species pairs."""

    if value is None:
        return ()
    return tuple((str(pair[0]), str(pair[1])) for pair in value)


@dataclass(frozen=True, slots=True)
class PerturbationTask:
    """A deterministic unit of worker execution.

    ``Atoms`` is deliberately retained as the scientific input rather than a
    path or an implicit positional tuple.  ASE structures are pickleable, so
    this record is serializable for ``ProcessPoolExecutor`` while preserving
    the exact base identity and effective seed.

    ``family`` and the half-open slot window identify a bounded family batch.
    They are optional for compatibility with callers that still submit one
    complete base task directly to :func:`execute_perturbation_task`.
    """

    base: Any
    base_structure_id: str
    settings: PerturbationSettings
    counts: PerturbationCounts
    seed: int
    base_ordinal: int = 0
    family: str | None = None
    slot_start: int = 0
    slot_stop: int | None = None
    # Prepared once by the parent coordinator for batched execution.  Workers
    # copy this context before any family mutates it; it is not part of task
    # identity or scientific provenance.
    prepared_supercell: Any = field(default=None, compare=False, repr=False)
    # This is populated only by the parent coordinator.  It is intentionally
    # excluded from identity/equality: the scientific task contract is the
    # base/family/slot tuple, while the queue is an execution detail.
    progress_queue: Any = field(default=None, compare=False, repr=False)

    @property
    def task_key(self) -> tuple[str, str, int, int | None]:
        """Return the historical base/family/slot identity."""

        return (self.base_structure_id, self.family or "all", self.slot_start, self.slot_stop)

    @property
    def progress_key(self) -> tuple[int, str, str, int, int | None]:
        """Return the collision-free parent-owned progress identity."""

        return (
            self.base_ordinal,
            self.base_structure_id,
            self.family or "all",
            self.slot_start,
            self.slot_stop,
        )


@dataclass(frozen=True, slots=True)
class PerturbationProgressEvent:
    """A bounded, process-safe worker progress notification.

    Counts are only populated when the worker has a confirmed scientific
    milestone.  In particular, a generator that is one indivisible operation
    reports a running phase rather than an invented percentage.
    """

    task_key: tuple[int, str, str, int, int | None]
    base_ordinal: int
    base_structure_id: str
    family: str
    slot_start: int
    slot_stop: int | None
    phase: str
    completed_units: int | None = None
    requested_units: int | None = None
    timestamp: float = 0.0
    worker_pid: int | None = None
    detail: str | None = None


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
    "LIQUID_FIDELITY",
    "LIQUID_METHOD",
    "PerturbationCounts",
    "PerturbationProgressEvent",
    "PerturbationSettings",
    "PerturbationTask",
    "PerturbationTaskResult",
    "PerturbationRejection",
    "PERTURBATION_FAMILY_SOURCE_FIELDS",
    "derive_child_seed",
]
