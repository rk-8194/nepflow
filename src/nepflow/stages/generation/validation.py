"""Validation specific to the generation phase's typed inputs."""

from __future__ import annotations

import math

from nepflow.config.models import CompositionConfig, GenerationConfig
from nepflow.errors import ConfigurationError


def validate_composition_config(config: CompositionConfig) -> CompositionConfig:
    """Validate the supported unary/binary/ternary composition space."""

    if not config.elements:
        raise ConfigurationError("generation requires at least one composition element")
    if len(config.elements) > 3:
        raise ConfigurationError(
            "generation supports unary, binary, and ternary compositions only"
        )
    if not math.isfinite(config.composition_step) or not 0.0 < config.composition_step <= 1.0:
        raise ConfigurationError("composition.composition_step must be in (0, 1]")
    steps = round(1.0 / config.composition_step)
    if steps < 1 or not math.isclose(
        steps * config.composition_step,
        1.0,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ConfigurationError(
            "composition.composition_step must divide the unit interval exactly"
        )
    if set(config.elements) & set(config.gas_elements):
        raise ConfigurationError("composition.elements and gas_elements must be disjoint")
    return config


def validate_generation_config(config: GenerationConfig) -> GenerationConfig:
    """Validate generator-facing settings without changing their defaults."""

    if not config.crystal_structures:
        raise ConfigurationError("generation.crystal_structures must not be empty")
    if config.target_n_atoms <= 0:
        raise ConfigurationError("generation.target_n_atoms must be positive")
    if config.n_workers < 0:
        raise ConfigurationError("generation.n_workers must be non-negative")
    return config
