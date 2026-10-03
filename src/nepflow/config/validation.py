"""Validation rules for the canonical NEPFlow configuration model."""

import math
import re

from ase.data import chemical_symbols

from nepflow.errors import ConfigurationError

from .models import NepflowConfig


ALLOWED_CRYSTAL_STRUCTURES = frozenset(
    {"bcc", "fcc", "hcp", "diamond", "simple_cubic"}
)
# Atomic representations do not yet have a structure-level selection result
# contract.  Reject the nominal mode instead of allowing it to fail later in
# descriptor caching or FPS row-to-structure mapping.
ALLOWED_DESCRIPTOR_TYPES = frozenset({"structure"})
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
        raise ConfigurationError(
            "project.config_version must match the root schema_version"
        )

    if require_scientific_fields and not config.composition.elements:
        raise ConfigurationError("Required configuration composition.elements is missing")
    if require_scientific_fields and not config.generation.crystal_structures:
        raise ConfigurationError(
            "Required configuration generation.crystal_structures is missing"
        )

    if config.composition.elements and not all(
        _is_element_symbol(element) for element in config.composition.elements
    ):
        raise ConfigurationError("composition.elements contains an invalid element symbol")
    if config.composition.gas_elements and not all(
        _is_element_symbol(element) for element in config.composition.gas_elements
    ):
        raise ConfigurationError(
            "composition.gas_elements contains an invalid element symbol"
        )
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
    _require_non_negative(
        "generation.target_n_atoms", config.generation.target_n_atoms, strictly_positive=True
    )
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
        "n_vacancy_interstitial",
        "n_gas_in_vacancy",
        "liquid_equilibration_steps",
        "liquid_steps_between_snapshots",
        "max_gas_occupancy",
    ):
        _require_non_negative(
            f"generation.{field_name}", getattr(config.generation, field_name)
        )
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
    ):
        _require_finite_non_negative(
            f"generation.{field_name}", getattr(config.generation, field_name)
        )
    _require_range(
        "generation.vacancy", config.generation.vacancy_min, config.generation.vacancy_max
    )
    _require_range(
        "generation.interstitial",
        config.generation.interstitial_min,
        config.generation.interstitial_max,
    )
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
        raise ConfigurationError(
            "generation.elastic_strain_amplitudes must contain finite values"
        )

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
    ):
        _require_non_negative(
            f"selection.{field_name}", getattr(config.selection, field_name), strictly_positive=field_name not in {"target_tolerance"}
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
    _require_non_negative(
        "dft_recovery.max_retry_level", config.dft_recovery.max_retry_level
    )
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
        raise ConfigurationError(
            "train_nep.weights count must match the configured element count"
        )
    if not all(
        math.isfinite(weight) and weight >= 0.0 for weight in config.train_nep.weights
    ):
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
        _require_non_negative(f"hpc.{field_name}", getattr(config.hpc, field_name), strictly_positive=True)
    for field_name in (
        "max_concurrent",
        "poll_interval",
        "gpumd_nodes",
        "gpumd_gpus",
        "memory_poll_interval",
    ):
        _require_non_negative(f"slurm.{field_name}", getattr(config.slurm, field_name), strictly_positive=True)
    _validate_time("slurm.walltime", config.slurm.walltime)
    _validate_time("slurm.train_nep_walltime", config.slurm.train_nep_walltime)
    _validate_time("slurm.gpumd_walltime", config.slurm.gpumd_walltime)
    _validate_time("slurm.memory_walltime", config.slurm.memory_walltime)

    return config


def _is_element_symbol(value: str) -> bool:
    return value in KNOWN_ELEMENT_SYMBOLS


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
