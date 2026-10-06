"""Validation specific to the generation phase's typed inputs."""

from __future__ import annotations

import math

from nepflow.config.models import (
    ALL_SOURCES,
    GENERATION_SOURCE_SCOPE_FIELDS,
    SUPPORTED_CONFIGURATIONAL_SOURCES,
    CompositionConfig,
    GenerationConfig,
)
from nepflow.errors import ConfigurationError


def validate_composition_config(config: CompositionConfig) -> CompositionConfig:
    """Validate the supported unary/binary/ternary composition space."""

    if not config.elements:
        raise ConfigurationError("generation requires at least one composition element")
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
    _validate_source_scopes(config)
    return config


def _validate_source_scopes(config: GenerationConfig) -> None:
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
