"""Controlled, deterministic candidate generation for training campaigns.

This module intentionally stops at a declarative matrix.  It provides the
stable trial contract that a future optimiser can consume without making any
adaptive or Bayesian search decision in the current issue.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import product
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from nepflow.config.models import NepTrainingConfig
from nepflow.io.hashing import sha256_bytes
from nepflow.io.json import canonical_json_bytes

_SCIENTIFIC_FIELDS = frozenset(
    {
        "population",
        "batch",
        "generations",
        "charge_mode",
        "weights",
        "outer_zbl",
        "cutoff",
        "n_max",
        "basis_size",
        "l_max",
        "neuron",
        "lambda_e",
        "lambda_f",
        "lambda_v",
        "lambda_shear",
    }
)


def _normalise_value(field_name: str, value: Any, current: Any) -> Any:
    """Convert sweep values to the exact typed configuration representation."""

    if isinstance(current, tuple):
        if isinstance(value, str):
            values = tuple(item for item in value.replace(",", " ").split() if item)
        else:
            values = tuple(value)
        if field_name == "weights":
            return tuple(float(item) for item in values)
        return tuple(str(item) for item in values)
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(f"Sweep value for {field_name} must be boolean")
        return value
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


@dataclass(frozen=True, slots=True)
class CandidateConfiguration:
    """One immutable effective NEP configuration in a controlled sweep."""

    ordinal: int
    hyperparameters: NepTrainingConfig
    overrides: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))

    @property
    def candidate_key(self) -> str:
        payload = {
            "hyperparameters": {
                field.name: getattr(self.hyperparameters, field.name)
                for field in fields(self.hyperparameters)
                if field.name != "sweep"
            },
        }
        return "candidate_" + sha256_bytes(canonical_json_bytes(payload))[:16]


@dataclass(frozen=True, slots=True)
class ControlledSweep:
    """Deterministic Cartesian candidate matrix over selected NEP fields."""

    base: NepTrainingConfig
    values: Mapping[str, tuple[Any, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            MappingProxyType({str(name): tuple(options) for name, options in self.values.items()}),
        )
        known = {field.name for field in fields(self.base)}
        unknown = sorted(set(self.values) - known)
        if unknown:
            raise ValueError("Unknown NEP sweep parameters: " + ", ".join(unknown))
        non_scientific = sorted(set(self.values) - _SCIENTIFIC_FIELDS)
        if non_scientific:
            raise ValueError(
                "Training sweeps may vary only scientific NEP settings: "
                + ", ".join(non_scientific)
            )
        for name, candidates in self.values.items():
            if not candidates:
                raise ValueError(f"Sweep parameter {name!r} has no candidate values")

    @classmethod
    def from_mapping(
        cls,
        base: NepTrainingConfig,
        values: Mapping[str, Sequence[Any]] | None = None,
    ) -> "ControlledSweep":
        return cls(
            base=base,
            values={
                str(name): tuple(options)
                for name, options in ({} if values is None else values).items()
            },
        )

    def configurations(self) -> tuple[CandidateConfiguration, ...]:
        if not self.values:
            return (CandidateConfiguration(0, self.base, {}),)
        names = tuple(sorted(self.values))
        combinations: list[CandidateConfiguration] = []
        for ordinal, selected in enumerate(product(*(self.values[name] for name in names))):
            overrides = {
                name: _normalise_value(name, value, getattr(self.base, name))
                for name, value in zip(names, selected)
            }
            configuration = replace(self.base, **overrides)
            combinations.append(CandidateConfiguration(ordinal, configuration, overrides))
        return tuple(combinations)

    def generate(self) -> tuple[CandidateConfiguration, ...]:
        """Alias used by callers that treat the sweep as a proposal source."""

        return self.configurations()


CandidateSweep = ControlledSweep
ControlledCandidateSweep = ControlledSweep


def generate_candidate_matrix(
    base: NepTrainingConfig,
    values: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[CandidateConfiguration, ...]:
    """Return a stable candidate matrix without consulting prior outcomes."""

    return ControlledSweep.from_mapping(base, values).configurations()


__all__ = [
    "CandidateConfiguration",
    "CandidateSweep",
    "ControlledCandidateSweep",
    "ControlledSweep",
    "generate_candidate_matrix",
]
