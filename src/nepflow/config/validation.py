"""Validation rules for the canonical NEPFlow configuration model."""

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ase.data import chemical_symbols

from nepflow.errors import ConfigurationError

from .models import (
    ALL_SOURCES,
    GENERATION_SOURCE_SCOPE_FIELDS,
    MAGNETIC_DEFECT_FAMILIES,
    SUPPORTED_CONFIGURATIONAL_SOURCES,
    SUPPORTED_SURFACE_MILLER_INDICES,
    GenerationConfig,
    NepflowConfig,
)

ALLOWED_CRYSTAL_STRUCTURES = frozenset({"bcc", "fcc", "hcp", "diamond", "simple_cubic"})
# Atomic representations do not yet have a structure-level selection result
# contract.  Reject the nominal mode instead of allowing it to fail later in
# descriptor caching or FPS row-to-structure mapping.
ALLOWED_DESCRIPTOR_TYPES = frozenset({"structure"})
ALLOWED_SELECTION_ALGORITHMS = frozenset({"fps", "information_entropy"})
ALLOWED_LOCAL_MAGNETIC_MODES = frozenset({"structural", "non_soc"})
TIME_PATTERN = re.compile(r"^(?:\d+):[0-5]\d:[0-5]\d$")
KNOWN_ELEMENT_SYMBOLS = frozenset(symbol for symbol in chemical_symbols if symbol)


def identity(value: str) -> str:
    """Return a prompt value unchanged."""
    return value


def normalize_element_list(raw: str, *, allow_blank: bool) -> str:
    """Normalize and validate a comma-separated list of element symbols."""
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        if allow_blank:
            return ""
        raise ValueError("At least one element is required")

    normalized: list[str] = []
    for item in items:
        symbol = item.capitalize()
        if symbol not in KNOWN_ELEMENT_SYMBOLS:
            raise ValueError(f"Unknown element: {item}")
        normalized.append(symbol)
    return ",".join(normalized)


def normalize_required_elements(raw: str) -> str:
    """Normalize a required comma-separated element list."""
    return normalize_element_list(raw, allow_blank=False)


def normalize_optional_elements(raw: str) -> str:
    """Normalize an optional comma-separated element list."""
    return normalize_element_list(raw, allow_blank=True)


def normalize_crystal_structures(raw: str) -> str:
    """Normalize and validate a comma-separated structure list."""
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one crystal structure is required")

    normalized: list[str] = []
    for item in items:
        if item not in ALLOWED_CRYSTAL_STRUCTURES:
            raise ValueError(f"Unknown crystal structure: {item}")
        normalized.append(item)
    return ",".join(normalized)


def normalize_target_n_atoms(raw: str) -> str:
    """Normalize target_n_atoms, defaulting to 128 when blank."""
    value = raw.strip()
    if not value:
        return "128"
    try:
        target = int(value)
    except ValueError as exc:
        raise ValueError("target_n_atoms must be a positive integer") from exc
    if target <= 0:
        raise ValueError("target_n_atoms must be a positive integer")
    return str(target)


def validate_config(
    config: NepflowConfig,
    *,
    require_scientific_fields: bool = True,
) -> NepflowConfig:
    """Validate a typed config and return it unchanged when valid.

    ``require_scientific_fields=False`` is used only by the temporary legacy
    stage adapter while individual Phase 4 stages still use partial fixtures.
    The direct loader defaults to complete project validation.
    """
    if config.schema_version != 1:
        raise ConfigurationError(
            f"Unsupported configuration schema_version={config.schema_version}; expected 1"
        )
    if config.project.config_version != config.schema_version:
        raise ConfigurationError("project.config_version must match the root schema_version")

    if require_scientific_fields and not config.composition.elements:
        raise ConfigurationError("Required configuration composition.elements is missing")
    if require_scientific_fields and not config.generation.crystal_structures:
        raise ConfigurationError("Required configuration generation.crystal_structures is missing")

    if config.composition.elements and not all(
        _is_element_symbol(element) for element in config.composition.elements
    ):
        raise ConfigurationError("composition.elements contains an invalid element symbol")
    if config.composition.gas_elements and not all(
        _is_element_symbol(element) for element in config.composition.gas_elements
    ):
        raise ConfigurationError("composition.gas_elements contains an invalid element symbol")
    if not 0.0 < config.composition.composition_step <= 1.0:
        raise ConfigurationError("composition.composition_step must be in (0, 1]")
    composition_steps = round(1.0 / config.composition.composition_step)
    if composition_steps < 1 or not math.isclose(
        composition_steps * config.composition.composition_step,
        1.0,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ConfigurationError(
            "composition.composition_step must divide the unit interval exactly"
        )
    if set(config.composition.elements) & set(config.composition.gas_elements):
        raise ConfigurationError("composition.elements and gas_elements must be disjoint")

    unknown_structures = set(config.generation.crystal_structures) - ALLOWED_CRYSTAL_STRUCTURES
    if unknown_structures:
        raise ConfigurationError(
            "generation.crystal_structures contains unsupported values: "
            + ", ".join(sorted(unknown_structures))
        )
    _validate_generation_source_scopes(config.generation)
    _validate_magnetism_settings(config)
    _require_non_negative(
        "generation.target_n_atoms", config.generation.target_n_atoms, strictly_positive=True
    )
    if (
        not math.isfinite(config.generation.composition_tolerance)
        or not 0.0 <= config.generation.composition_tolerance <= 1.0
    ):
        raise ConfigurationError("generation.composition_tolerance must be in [0, 1]")
    _require_non_negative("generation.n_workers", config.generation.n_workers)
    for field_name in (
        "n_random_solid_solution",
        "n_sqs",
        "n_segregated",
        "n_liquid_configurations",
        "n_liquid_snapshots",
        "n_volume_points",
        "n_rattled",
        "n_vacancies",
        "n_interstitials",
        "n_gas_interstitials",
        "n_substitutions",
        "n_antisites",
        "n_vacancy_interstitial",
        "n_gas_in_vacancy",
        "n_surfaces",
        "n_grain_boundaries",
        "liquid_equilibration_steps",
        "liquid_steps_between_snapshots",
        "max_gas_occupancy",
        "interstitial_max_attempts",
    ):
        _require_non_negative(f"generation.{field_name}", getattr(config.generation, field_name))
    for field_name in (
        "liquid_temperature",
        "liquid_timestep_fs",
        "liquid_friction",
        "rattle_std",
        "rattle_std_min",
        "rattle_std_max",
        "rattle_d_min",
        "interstitial_d_min",
        "gas_interstitial_d_min",
        "defect_defect_d_min",
        "periodic_image_d_min",
        "substitution_min",
        "substitution_max",
        "antisite_min",
        "antisite_max",
        "surface_vacuum",
        "surface_min_in_plane_dimensions",
        "grain_boundary_misorientation_angle",
        "grain_boundary_min_thickness",
        "grain_boundary_overlap_tolerance",
    ):
        if field_name == "surface_min_in_plane_dimensions":
            values = config.generation.surface_min_in_plane_dimensions
            if len(values) != 2 or any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ConfigurationError(
                    "generation.surface_min_in_plane_dimensions must contain two finite non-negative values"
                )
            continue
        _require_finite_non_negative(
            f"generation.{field_name}", getattr(config.generation, field_name)
        )
    if config.generation.use_liquid and (
        config.generation.n_liquid_configurations > 0 and config.generation.n_liquid_snapshots > 0
    ):
        if config.generation.liquid_timestep_fs <= 0.0:
            raise ConfigurationError("generation.liquid_timestep_fs must be positive")
        if config.generation.liquid_steps_between_snapshots <= 0:
            raise ConfigurationError("generation.liquid_steps_between_snapshots must be positive")
    _validate_surface_settings(config.generation)
    _validate_grain_boundary_settings(config.generation)
    _require_range(
        "generation.vacancy", config.generation.vacancy_min, config.generation.vacancy_max
    )
    _require_range(
        "generation.interstitial",
        config.generation.interstitial_min,
        config.generation.interstitial_max,
    )
    _require_range(
        "generation.substitution",
        config.generation.substitution_min,
        config.generation.substitution_max,
    )
    _require_range(
        "generation.antisite",
        config.generation.antisite_min,
        config.generation.antisite_max,
    )
    _validate_point_defect_settings(config.generation)
    _require_range(
        "generation.volume_scale",
        config.generation.volume_scale_min,
        config.generation.volume_scale_max,
        strictly_positive=True,
    )
    _require_range(
        "generation.rattle_std",
        config.generation.rattle_std_min,
        config.generation.rattle_std_max,
    )
    if not config.generation.elastic_strain_amplitudes:
        raise ConfigurationError(
            "generation.elastic_strain_amplitudes must contain at least one value"
        )
    if not all(math.isfinite(value) for value in config.generation.elastic_strain_amplitudes):
        raise ConfigurationError("generation.elastic_strain_amplitudes must contain finite values")

    if config.selection.algorithm not in ALLOWED_SELECTION_ALGORITHMS:
        raise ConfigurationError(
            "selection.algorithm must be one of: " + ", ".join(sorted(ALLOWED_SELECTION_ALGORITHMS))
        )
    entropy = config.selection.entropy
    if not math.isfinite(entropy.beta) or entropy.beta <= 0.0:
        raise ConfigurationError("selection.entropy.beta must be finite and positive")
    if entropy.optimizer_method not in {"lazy_greedy", "full_greedy"}:
        raise ConfigurationError(
            "selection.entropy.optimizer_method must be lazy_greedy or full_greedy"
        )
    for field_name in ("local_cutoff", "whitening_tolerance"):
        value = getattr(entropy, field_name)
        if not math.isfinite(value) or value <= 0.0:
            raise ConfigurationError(f"selection.entropy.{field_name} must be finite and positive")
    if not isinstance(entropy.local_radial_bins, int) or entropy.local_radial_bins < 1:
        raise ConfigurationError("selection.entropy.local_radial_bins must be positive")
    if not isinstance(entropy.local_angular_bins, int) or entropy.local_angular_bins < 1:
        raise ConfigurationError("selection.entropy.local_angular_bins must be positive")
    if (
        not math.isfinite(entropy.whitening_regularization)
        or entropy.whitening_regularization < 0.0
    ):
        raise ConfigurationError(
            "selection.entropy.whitening_regularization must be finite and non-negative"
        )
    if entropy.whitening_singular_policy not in {"drop", "regularize", "reject"}:
        raise ConfigurationError(
            "selection.entropy.whitening_singular_policy must be drop, regularize, or reject"
        )
    for field_name in ("local_radial_sigma", "local_angular_sigma"):
        value = getattr(entropy, field_name)
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ConfigurationError(f"selection.entropy.{field_name} must be finite and positive")
    bandwidth = entropy.bandwidth
    if bandwidth.mode not in {"automatic", "manual"}:
        raise ConfigurationError("selection.entropy.bandwidth.mode must be automatic or manual")
    if bandwidth.backend not in {"exact_cpu", "exact_indexed_cpu"}:
        raise ConfigurationError(
            "selection.entropy.bandwidth.backend must be exact_cpu or exact_indexed_cpu"
        )
    if bandwidth.metric != "euclidean":
        raise ConfigurationError("selection.entropy.bandwidth.metric only supports euclidean")
    if not isinstance(bandwidth.chunk_size, int) or bandwidth.chunk_size < 1:
        raise ConfigurationError("selection.entropy.bandwidth.chunk_size must be positive")
    if not isinstance(bandwidth.max_neighbour_entries, int) or bandwidth.max_neighbour_entries < 1:
        raise ConfigurationError(
            "selection.entropy.bandwidth.max_neighbour_entries must be positive"
        )
    if bandwidth.max_index_bytes is not None and (
        not isinstance(bandwidth.max_index_bytes, int) or bandwidth.max_index_bytes < 1
    ):
        raise ConfigurationError(
            "selection.entropy.bandwidth.max_index_bytes must be positive when provided"
        )
    if (
        not isinstance(bandwidth.max_radius_query_bytes, int)
        or bandwidth.max_radius_query_bytes < 1
    ):
        raise ConfigurationError(
            "selection.entropy.bandwidth.max_radius_query_bytes must be positive"
        )
    if (
        not isinstance(bandwidth.max_calibration_work_bytes, int)
        or bandwidth.max_calibration_work_bytes < 1
    ):
        raise ConfigurationError(
            "selection.entropy.bandwidth.max_calibration_work_bytes must be positive"
        )
    if bandwidth.calibration_batch_size is not None and (
        not isinstance(bandwidth.calibration_batch_size, int)
        or bandwidth.calibration_batch_size < 1
    ):
        raise ConfigurationError(
            "selection.entropy.bandwidth.calibration_batch_size must be positive when provided"
        )
    if tuple(sorted(set(bandwidth.k_candidates))) != tuple(bandwidth.k_candidates) or any(
        value < 1 for value in bandwidth.k_candidates
    ):
        raise ConfigurationError("selection.entropy.bandwidth.k_candidates must be increasing")
    if tuple(sorted(set(bandwidth.c_candidates))) != tuple(bandwidth.c_candidates) or any(
        value <= 0.0 or not math.isfinite(value) for value in bandwidth.c_candidates
    ):
        raise ConfigurationError("selection.entropy.bandwidth.c_candidates must be increasing")
    if bandwidth.mode == "manual":
        if bandwidth.k is None or bandwidth.k < 1:
            raise ConfigurationError("manual entropy bandwidth requires positive k")
        if bandwidth.c is None or not math.isfinite(bandwidth.c) or bandwidth.c <= 0.0:
            raise ConfigurationError("manual entropy bandwidth requires positive c")
    elif bandwidth.k is not None or bandwidth.c is not None:
        raise ConfigurationError("automatic entropy bandwidth cannot specify manual k or c")
    for field_name in ("max_edges", "max_entries"):
        value = getattr(entropy, field_name)
        if not isinstance(value, int) or value < 1:
            raise ConfigurationError(f"selection.entropy.{field_name} must be positive")
    for field_name in (
        "max_graph_bytes",
        "max_graph_spool_bytes",
        "max_contribution_bytes",
        "max_contribution_spool_bytes",
    ):
        value = getattr(entropy, field_name)
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ConfigurationError(f"selection.entropy.{field_name} must be positive")
    if any(not str(species).strip() for species in entropy.local_species):
        raise ConfigurationError("selection.entropy.local_species must not contain blanks")
    if config.selection.local_magnetic_mode not in ALLOWED_LOCAL_MAGNETIC_MODES:
        raise ConfigurationError(
            "selection.local_magnetic_mode must be one of: "
            + ", ".join(sorted(ALLOWED_LOCAL_MAGNETIC_MODES))
        )
    if isinstance(config.selection.local_descriptor_workers, bool) or not isinstance(
        config.selection.local_descriptor_workers, int
    ):
        raise ConfigurationError("selection.local_descriptor_workers must be an integer")
    if (
        isinstance(config.selection.max_local_descriptor_inflight_bytes, bool)
        or not isinstance(config.selection.max_local_descriptor_inflight_bytes, int)
        or config.selection.max_local_descriptor_inflight_bytes < 1
    ):
        raise ConfigurationError(
            "selection.max_local_descriptor_inflight_bytes must be a positive integer"
        )
    if (
        not math.isfinite(config.selection.background_mass)
        or config.selection.background_mass <= 0.0
    ):
        raise ConfigurationError("selection.background_mass must be finite and positive")
    if config.selection.descriptor_type not in ALLOWED_DESCRIPTOR_TYPES:
        raise ConfigurationError(
            "selection.descriptor_type must be one of: "
            + ", ".join(sorted(ALLOWED_DESCRIPTOR_TYPES))
        )
    for field_name in (
        "batch_size",
        "target_train_count",
        "target_test_count",
        "target_tolerance",
        "max_search_iterations",
        "composition_aware_fps_adaptive_retries",
        "local_descriptor_workers",
        "max_local_descriptor_inflight_bytes",
    ):
        _require_non_negative(
            f"selection.{field_name}",
            getattr(config.selection, field_name),
            strictly_positive=field_name not in {"target_tolerance", "local_descriptor_workers"},
        )
    for field_name in (
        "test_pool_factor",
        "composition_aware_fps_frontier_fraction",
        "composition_aware_fps_descriptor_floor_fraction",
    ):
        value = getattr(config.selection, field_name)
        if not 0.0 < value <= 1.0:
            raise ConfigurationError(f"selection.{field_name} must be in (0, 1]")
    if config.selection.composition_aware_fps_ternary_weight < 0.0:
        raise ConfigurationError(
            "selection.composition_aware_fps_ternary_weight must be non-negative"
        )

    if config.vasp.kspacing <= 0.0 or not math.isfinite(config.vasp.kspacing):
        raise ConfigurationError("vasp.kspacing must be a positive finite float")
    _require_non_negative("dft_recovery.max_retry_level", config.dft_recovery.max_retry_level)
    _validate_time("dft_recovery.vasp_walltime", config.dft_recovery.vasp_walltime)

    if config.train_nep.charge_mode not in (0, 1):
        raise ConfigurationError("train_nep.charge_mode must be 0 or 1")
    for field_name in (
        "population",
        "batch",
        "generations",
        "max_resubmit",
    ):
        _require_non_negative(
            f"train_nep.{field_name}", getattr(config.train_nep, field_name), strictly_positive=True
        )
    if config.train_nep.weights and len(config.train_nep.weights) != len(
        config.composition.elements + config.composition.gas_elements
    ):
        raise ConfigurationError("train_nep.weights count must match the configured element count")
    if not all(math.isfinite(weight) and weight >= 0.0 for weight in config.train_nep.weights):
        raise ConfigurationError("train_nep.weights must be finite and non-negative")
    for field_name in (
        "outer_zbl",
        "lambda_e",
        "lambda_f",
        "lambda_v",
        "lambda_shear",
    ):
        value = getattr(config.train_nep, field_name)
        if not math.isfinite(value) or value < 0.0:
            raise ConfigurationError(f"train_nep.{field_name} must be finite and non-negative")
    for field_name in ("cutoff", "n_max", "basis_size", "l_max", "neuron"):
        if not getattr(config.train_nep, field_name):
            raise ConfigurationError(f"train_nep.{field_name} must not be empty")

    for field_name in ("cores_per_node", "gpus_per_node", "max_nodes"):
        _require_non_negative(
            f"hpc.{field_name}", getattr(config.hpc, field_name), strictly_positive=True
        )
    for field_name in (
        "max_concurrent",
        "poll_interval",
        "gpumd_nodes",
        "gpumd_gpus",
        "memory_poll_interval",
    ):
        _require_non_negative(
            f"slurm.{field_name}", getattr(config.slurm, field_name), strictly_positive=True
        )
    _validate_time("slurm.walltime", config.slurm.walltime)
    _validate_time("slurm.train_nep_walltime", config.slurm.train_nep_walltime)
    _validate_time("slurm.gpumd_walltime", config.slurm.gpumd_walltime)
    _validate_time("slurm.memory_walltime", config.slurm.memory_walltime)

    return config


def _is_element_symbol(value: str) -> bool:
    return value in KNOWN_ELEMENT_SYMBOLS


def _validate_point_defect_settings(config: GenerationConfig) -> None:
    """Validate species and crystallographic inputs used by defect families."""

    for field_name in ("vacancy_species",):
        values = getattr(config, field_name)
        if any(not _is_element_symbol(value) for value in values):
            raise ConfigurationError(f"generation.{field_name} contains an invalid element symbol")
    for field_name in ("substitution_pairs", "antisite_pairs"):
        pairs = getattr(config, field_name)
        for pair in pairs:
            if len(pair) != 2 or any(not _is_element_symbol(value) for value in pair):
                raise ConfigurationError(
                    f"generation.{field_name} contains an invalid element pair"
                )
            if pair[0] == pair[1]:
                raise ConfigurationError(
                    f"generation.{field_name} pairs must contain distinct elements"
                )
    for field_name in ("interstitial_sites", "crystallographic_interstitial_sites"):
        for site in getattr(config, field_name):
            coordinates: Sequence[Any] | None
            if isinstance(site, Mapping):
                raw = site.get("fractional", site.get("position"))
                coordinates = (
                    raw if isinstance(raw, Sequence) and not isinstance(raw, str) else None
                )
            else:
                coordinates = (
                    site if isinstance(site, Sequence) and not isinstance(site, str) else None
                )
            if coordinates is None or len(coordinates) != 3:
                raise ConfigurationError(
                    f"generation.{field_name} sites must contain three coordinates"
                )
            try:
                values = tuple(float(value) for value in coordinates)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"generation.{field_name} sites must contain numeric coordinates"
                ) from exc
            if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in values):
                raise ConfigurationError(
                    f"generation.{field_name} sites must use fractional coordinates in [0, 1)"
                )


def _validate_surface_settings(config: GenerationConfig) -> None:
    """Validate Miller, slab, termination, and in-plane surface controls."""

    if config.surface_thickness is not None and (
        not math.isfinite(config.surface_thickness) or config.surface_thickness <= 0.0
    ):
        raise ConfigurationError("generation.surface_thickness must be positive and finite")
    _require_non_negative("generation.surface_layers", config.surface_layers)
    if config.surface_thickness is None and config.surface_layers <= 0:
        raise ConfigurationError(
            "generation.surface_layers must be positive when surface_thickness is not set"
        )
    if not math.isfinite(config.surface_min_half_depth) or config.surface_min_half_depth <= 0.0:
        raise ConfigurationError("generation.surface_min_half_depth must be positive and finite")
    if (
        not math.isfinite(config.surface_bulk_environment_radius)
        or config.surface_bulk_environment_radius <= 0.0
    ):
        raise ConfigurationError(
            "generation.surface_bulk_environment_radius must be positive and finite"
        )
    if config.surface_min_bulk_core_atoms <= 0:
        raise ConfigurationError("generation.surface_min_bulk_core_atoms must be positive")
    if (
        not math.isfinite(config.surface_bulk_environment_distance_tolerance)
        or config.surface_bulk_environment_distance_tolerance < 0.0
    ):
        raise ConfigurationError(
            "generation.surface_bulk_environment_distance_tolerance must be finite and non-negative"
        )
    if config.surface_target_n_atoms is not None and (
        isinstance(config.surface_target_n_atoms, bool) or config.surface_target_n_atoms <= 0
    ):
        raise ConfigurationError("generation.surface_target_n_atoms must be positive when set")
    if not math.isfinite(config.surface_target_tolerance) or config.surface_target_tolerance < 0.0:
        raise ConfigurationError(
            "generation.surface_target_tolerance must be finite and non-negative"
        )
    if (
        isinstance(config.surface_max_n_atoms, bool)
        or not isinstance(config.surface_max_n_atoms, int)
        or config.surface_max_n_atoms <= 0
    ):
        raise ConfigurationError("generation.surface_max_n_atoms must be positive")
    _require_non_negative("generation.surface_max_terminations", config.surface_max_terminations)
    if config.surface_termination_policy not in {"all", "first"}:
        raise ConfigurationError("generation.surface_termination_policy must be 'all' or 'first'")
    if not config.surface_miller_indices:
        if config.surface_enabled or config.n_surfaces:
            raise ConfigurationError(
                "generation.surface_miller_indices must contain at least one index"
            )
    for index in config.surface_miller_indices:
        if len(index) != 3 or any(
            not isinstance(value, int) or isinstance(value, bool) for value in index
        ):
            raise ConfigurationError(
                "generation.surface_miller_indices must contain integer triples"
            )
        if not any(index):
            raise ConfigurationError("generation.surface_miller_indices cannot contain (0, 0, 0)")
        if tuple(index) not in SUPPORTED_SURFACE_MILLER_INDICES:
            supported = ", ".join(
                f"({h},{k},{miller_l})"
                for h, k, miller_l in sorted(SUPPORTED_SURFACE_MILLER_INDICES)
            )
            raise ConfigurationError(
                "generation.surface_miller_indices contains unsupported orientation "
                f"{tuple(index)}; supported orientations are {supported}"
            )
    if len(config.surface_in_plane_repeat) != 2 or any(
        not isinstance(value, int) or value <= 0 for value in config.surface_in_plane_repeat
    ):
        raise ConfigurationError(
            "generation.surface_in_plane_repeat must contain two positive integers"
        )
    if len(config.surface_max_in_plane_repeat) != 2 or any(
        not isinstance(value, int) or value <= 0 for value in config.surface_max_in_plane_repeat
    ):
        raise ConfigurationError(
            "generation.surface_max_in_plane_repeat must contain two positive integers"
        )
    if any(
        lower > upper
        for lower, upper in zip(config.surface_in_plane_repeat, config.surface_max_in_plane_repeat)
    ):
        raise ConfigurationError(
            "generation.surface_max_in_plane_repeat must not be below surface_in_plane_repeat"
        )
    if (
        isinstance(config.surface_max_normal_repeat, bool)
        or not isinstance(config.surface_max_normal_repeat, int)
        or config.surface_max_normal_repeat <= 0
    ):
        raise ConfigurationError("generation.surface_max_normal_repeat must be positive")
    if len(config.surface_min_in_plane_dimensions) != 2 or any(
        not math.isfinite(value) or value < 0.0 for value in config.surface_min_in_plane_dimensions
    ):
        raise ConfigurationError(
            "generation.surface_min_in_plane_dimensions must be finite and non-negative"
        )
    if not math.isfinite(config.surface_vacuum) or config.surface_vacuum <= 0.0:
        raise ConfigurationError("generation.surface_vacuum must be positive and finite")
    if str(config.surface_stoichiometry_policy).strip().lower() not in {
        "allow",
        "reject",
        "reject_changed",
        "require_stoichiometric",
    }:
        raise ConfigurationError(
            "generation.surface_stoichiometry_policy must be 'allow' or 'reject'"
        )
    if str(config.surface_polarity_policy).strip().lower() not in {
        "allow",
        "reject_known_polar",
        "reject_polar",
        "require_known_nonpolar",
        "require_nonpolar",
    }:
        raise ConfigurationError(
            "generation.surface_polarity_policy must be 'allow', "
            "'reject_known_polar', or 'require_known_nonpolar'"
        )


def _validate_grain_boundary_settings(config: GenerationConfig) -> None:
    """Validate the initial supported Sigma-5 [001] tilt relationship."""

    axis = config.grain_boundary_rotation_axis
    plane = config.grain_boundary_plane
    if len(axis) != 3 or any(not isinstance(value, int) for value in axis) or not any(axis):
        raise ConfigurationError(
            "generation.grain_boundary_rotation_axis must be a non-zero integer triple"
        )
    if len(plane) != 3 or any(not isinstance(value, int) for value in plane) or not any(plane):
        raise ConfigurationError(
            "generation.grain_boundary_plane must be a non-zero integer triple"
        )
    if tuple(axis) != (0, 0, 1):
        raise ConfigurationError(
            "only the supported Sigma-5 [001] grain-boundary axis is available"
        )
    if tuple(plane) != (2, 1, 0):
        raise ConfigurationError(
            "only the supported Sigma-5 (210) grain-boundary plane is available"
        )
    if config.grain_boundary_sigma != 5:
        raise ConfigurationError("only Sigma 5 grain boundaries are supported")
    if not math.isclose(
        config.grain_boundary_misorientation_angle,
        36.86989764584402,
        rel_tol=1.0e-9,
        abs_tol=1.0e-8,
    ):
        raise ConfigurationError(
            "only the 36.86989764584402 degree Sigma-5 misorientation is supported"
        )
    _require_non_negative(
        "generation.grain_boundary_expand_times",
        config.grain_boundary_expand_times,
        strictly_positive=True,
    )
    _require_non_negative(
        "generation.grain_boundary_sigma", config.grain_boundary_sigma, strictly_positive=True
    )
    if not 0.0 <= config.grain_boundary_overlap_tolerance <= 1.0:
        raise ConfigurationError("generation.grain_boundary_overlap_tolerance must be in [0, 1]")


def _validate_generation_source_scopes(config: GenerationConfig) -> None:
    for field_name in GENERATION_SOURCE_SCOPE_FIELDS:
        scope = getattr(config, field_name)
        name = f"generation.{field_name}"
        if not isinstance(scope, tuple) or not scope:
            raise ConfigurationError(f"{name} must contain at least one explicit source")
        if any(not isinstance(source, str) or not source.strip() for source in scope):
            raise ConfigurationError(f"{name} contains a blank source")
        normalized = tuple(source.strip().lower() for source in scope)
        if normalized != scope:
            raise ConfigurationError(f"{name} must use normalized lowercase source names")
        if len(set(scope)) != len(scope):
            raise ConfigurationError(f"{name} must not contain duplicate sources")
        if ALL_SOURCES in scope and len(scope) != 1:
            raise ConfigurationError(f"{name} cannot combine 'all' with named sources")
        unknown = set(scope) - SUPPORTED_CONFIGURATIONAL_SOURCES - {ALL_SOURCES}
        if unknown:
            raise ConfigurationError(
                f"{name} contains unsupported sources: {', '.join(sorted(unknown))}"
            )


def _validate_magnetism_settings(config: NepflowConfig) -> None:
    magnetic = config.magnetism
    if magnetic.enabled and not (
        magnetic.include_non_magnetic
        or magnetic.include_ferromagnetic
        or magnetic.include_antiferromagnetic
    ):
        raise ConfigurationError(
            "magnetism must include at least one of non-magnetic, FM, or AFM ordering"
        )
    if not magnetic.enabled and (
        magnetic.include_ferromagnetic or magnetic.include_antiferromagnetic
    ):
        raise ConfigurationError(
            "magnetism FM/AFM orderings cannot be enabled while magnetism.enabled is false"
        )
    if (magnetic.include_ferromagnetic or magnetic.include_antiferromagnetic) and not (
        magnetic.target_potential_magnetic
    ):
        raise ConfigurationError(
            "magnetism FM/AFM orderings require target-potential magnetic capability"
        )
    if magnetic.include_antiferromagnetic and not magnetic.moment_sets:
        raise ConfigurationError("magnetism.moment_sets is required when AFM ordering is enabled")
    if magnetic.include_ferromagnetic and not magnetic.moment_sets:
        raise ConfigurationError("magnetism.moment_sets is required when FM ordering is enabled")

    names = [moment_set.name for moment_set in magnetic.moment_sets]
    if len(set(names)) != len(names):
        raise ConfigurationError("magnetism.moment_sets must have unique names")
    element_sets = {frozenset(moment_set.element_moments) for moment_set in magnetic.moment_sets}
    if len(element_sets) > 1:
        raise ConfigurationError("magnetism.moment_sets must define the same complete element set")
    allowed_elements = set(config.composition.elements) | set(config.composition.gas_elements)
    for moment_set in magnetic.moment_sets:
        for element, magnitude in moment_set.element_moments.items():
            if not _is_element_symbol(element):
                raise ConfigurationError(
                    f"magnetism.moment_sets contains an invalid element symbol: {element}"
                )
            if allowed_elements and element not in allowed_elements:
                raise ConfigurationError(
                    f"magnetism.moment_sets element {element} is absent from composition.elements"
                )
            if not math.isfinite(magnitude) or magnitude <= 0.0:
                raise ConfigurationError(
                    f"magnetism.moment_sets magnitude for {element} must be finite and positive"
                )

    if not math.isfinite(magnetic.symmetry_tolerance) or magnetic.symmetry_tolerance <= 0.0:
        raise ConfigurationError("magnetism.symmetry_tolerance must be finite and positive")
    if not math.isfinite(magnetic.phase_tolerance) or magnetic.phase_tolerance <= 0.0:
        raise ConfigurationError("magnetism.phase_tolerance must be finite and positive")
    for field_name in (
        "max_afm_orderings",
        "max_magnetic_variants_per_parent",
        "max_magnetic_variants_per_defect",
    ):
        _require_non_negative(
            f"magnetism.{field_name}",
            getattr(magnetic, field_name),
            strictly_positive=True,
        )
    _require_non_negative("magnetism.max_defect_parents", magnetic.max_defect_parents)
    if magnetic.unmapped_site_policy not in {"skip_afm", "reject", "allow_fm_only"}:
        raise ConfigurationError(
            "magnetism.unmapped_site_policy must be 'skip_afm', 'reject', or 'allow_fm_only'"
        )
    _validate_magnetic_source_scope(magnetic.magnetic_sources)
    if len(set(magnetic.defect_families)) != len(magnetic.defect_families):
        raise ConfigurationError("magnetism.defect_families must not contain duplicates")
    unknown_families = set(magnetic.defect_families) - set(MAGNETIC_DEFECT_FAMILIES)
    if unknown_families:
        raise ConfigurationError(
            "magnetism.defect_families contains unsupported families: "
            + ", ".join(sorted(unknown_families))
        )


def _validate_magnetic_source_scope(scope: tuple[str, ...]) -> None:
    name = "magnetism.magnetic_sources"
    if not isinstance(scope, tuple) or not scope:
        raise ConfigurationError(f"{name} must contain at least one explicit source")
    if any(not isinstance(source, str) or not source.strip() for source in scope):
        raise ConfigurationError(f"{name} contains a blank source")
    normalized = tuple(source.strip().lower() for source in scope)
    if normalized != scope:
        raise ConfigurationError(f"{name} must use normalized lowercase source names")
    if len(set(scope)) != len(scope):
        raise ConfigurationError(f"{name} must not contain duplicate sources")
    if ALL_SOURCES in scope and len(scope) != 1:
        raise ConfigurationError(f"{name} cannot combine 'all' with named sources")
    unknown = set(scope) - SUPPORTED_CONFIGURATIONAL_SOURCES - {ALL_SOURCES}
    if unknown:
        raise ConfigurationError(
            f"{name} contains unsupported sources: {', '.join(sorted(unknown))}"
        )


def _require_non_negative(
    name: str,
    value: int,
    *,
    strictly_positive: bool = False,
) -> None:
    if (strictly_positive and value <= 0) or (not strictly_positive and value < 0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ConfigurationError(f"{name} must be {qualifier}")


def _require_finite_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ConfigurationError(f"{name} must be finite and non-negative")


def _require_range(
    name: str,
    minimum: float,
    maximum: float,
    *,
    strictly_positive: bool = False,
) -> None:
    if minimum > maximum:
        raise ConfigurationError(f"{name}_min must not exceed {name}_max")
    if strictly_positive and minimum <= 0.0:
        raise ConfigurationError(f"{name}_min must be positive")


def _validate_time(name: str, value: str | None) -> None:
    if value is not None and not TIME_PATTERN.fullmatch(value):
        raise ConfigurationError(f"{name} must use HH:MM:SS format")
