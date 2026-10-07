"""Cross-stage magnetic records and magnetic-state identities.

This module owns the stable scientific representation of a magnetic candidate.
Generation-specific enumeration requests stay in the generation stage; these
records are intentionally small enough to be consumed by generation,
selection, and later DFT adapters.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, cast

from nepflow.io.hashing import sha256_canonical_json

MAGNETIC_STATE_IDENTITY_SCHEMA = "magnetic-state-v1"
MAGNETIC_MOMENTS_ARRAY = "magnetic_moments"
MAGNETIC_CONSTRAINT_MASK_ARRAY = "magnetic_constraint_mask"
_ELEMENT_PATTERN = re.compile(r"^[A-Z][a-z]?$")


class MagneticOrdering(str, Enum):
    """Supported version-one collinear ordering labels."""

    NONMAGNETIC = "nonmagnetic"
    FERROMAGNETIC = "fm"
    ANTIFERROMAGNETIC = "afm"

    # Short names are part of the public typed contract.
    NON_MAGNETIC = "nonmagnetic"
    FERRO = "fm"
    ANTIFERRO = "afm"
    NM = "nonmagnetic"
    FM = "fm"
    AFM = "afm"

    @classmethod
    def parse(cls, value: "MagneticOrdering | str") -> "MagneticOrdering":
        if isinstance(value, cls):
            return value
        aliases = {
            "nm": cls.NONMAGNETIC,
            "nonmagnetic": cls.NONMAGNETIC,
            "non_magnetic": cls.NONMAGNETIC,
            "non-magnetic": cls.NONMAGNETIC,
            "fm": cls.FERROMAGNETIC,
            "ferromagnetic": cls.FERROMAGNETIC,
            "afm": cls.ANTIFERROMAGNETIC,
            "antiferromagnetic": cls.ANTIFERROMAGNETIC,
            "antiferro": cls.ANTIFERROMAGNETIC,
        }
        try:
            return aliases[str(value).strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported magnetic ordering: {value!r}") from exc


def canonical_atom_order(atoms: Any) -> tuple[int, ...]:
    """Return the atom order used by canonical_structure_text.

    Structure identity groups atoms by sorted chemical symbol and preserves
    input order within each element. Magnetic vectors are reordered with this
    same rule before they enter a magnetic identity.
    """

    symbols = tuple(str(symbol) for symbol in atoms.get_chemical_symbols())
    return tuple(sorted(range(len(symbols)), key=lambda index: (symbols[index], index)))


def _normalise_vector(value: Sequence[Any], *, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three components")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return (
        0.0 if result[0] == 0.0 else result[0],
        0.0 if result[1] == 0.0 else result[1],
        0.0 if result[2] == 0.0 else result[2],
    )


def _normalise_moments(
    moments: Any,
    constraint_mask: Sequence[Any],
    *,
    atom_order: Sequence[int] | None,
    name: str = "moments",
) -> tuple[tuple[float, float, float], ...]:
    try:
        raw_moments = tuple(_normalise_vector(row, name=name) for row in moments)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of three-component vectors") from exc
    if len(constraint_mask) != len(raw_moments):
        raise ValueError("constraint_mask length must match moments")
    normalized_mask = tuple(bool(value) for value in constraint_mask)
    if atom_order is not None:
        order = tuple(int(index) for index in atom_order)
        if sorted(order) != list(range(len(raw_moments))):
            raise ValueError("atom_order must be a permutation of the moment rows")
        raw_moments = tuple(raw_moments[index] for index in order)
        normalized_mask = tuple(normalized_mask[index] for index in order)
    return cast(tuple[tuple[float, float, float], ...], raw_moments)


def _canonicalize_global_inversion(
    moments: tuple[tuple[float, float, float], ...],
    *,
    enabled: bool,
) -> tuple[tuple[float, float, float], ...]:
    if not enabled:
        return moments
    inverted = tuple(
        tuple(0.0 if component == 0.0 else -component for component in vector) for vector in moments
    )
    return cast(tuple[tuple[float, float, float], ...], min(moments, inverted))


def _normalise_optional_vector(
    value: Sequence[Any] | None,
    *,
    name: str,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return _normalise_vector(value, name=name)


def _normalise_phases(value: Sequence[Any] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    phases = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in phases):
        raise ValueError("orbit_phases must contain finite values")
    return tuple(0.0 if item == 0.0 else item for item in phases)


def calculate_magnetic_state_id(
    moments: Any,
    constraint_mask: Sequence[Any],
    **kwargs: Any,
) -> str:
    """Calculate a magnetic-state identity from its canonical state fields."""

    return MagneticStateIdentity.from_components(
        moments,
        constraint_mask,
        **kwargs,
    ).magnetic_state_id


@dataclass(frozen=True, slots=True)
class MagneticMomentSet:
    """Named positive target moment magnitudes in Bohr magnetons."""

    name: str
    element_moments: Mapping[str, float] = field(default_factory=dict)
    # "moments" is accepted as a readable constructor alias for callers
    # using the compact vocabulary from the magnetic PDD.
    moments: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("magnetic moment-set name must not be blank")
        source = self.element_moments if self.moments is None else self.moments
        if not isinstance(source, Mapping) or not source:
            raise ValueError(f"magnetic moment set {name!r} must contain element moments")
        normalized: dict[str, float] = {}
        for element, magnitude in source.items():
            symbol = str(element).strip()
            if not _ELEMENT_PATTERN.fullmatch(symbol):
                raise ValueError(f"invalid element symbol in magnetic moment set: {element!r}")
            value = float(magnitude)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"magnetic moment for {symbol} in set {name!r} must be finite and positive"
                )
            if symbol in normalized:
                raise ValueError(f"duplicate element in magnetic moment set {name!r}: {symbol}")
            normalized[symbol] = value
        frozen = MappingProxyType(dict(sorted(normalized.items())))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "element_moments", frozen)
        object.__setattr__(self, "moments", frozen)

    @classmethod
    def from_mapping(cls, name: str, values: Mapping[str, Any]) -> "MagneticMomentSet":
        return cls(name=name, element_moments=values)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "moments": dict(self.element_moments)}


@dataclass(frozen=True, slots=True)
class MagneticStateIdentity:
    """Stable identity of one actual local-moment and constraint field."""

    moments: tuple[tuple[float, float, float], ...]
    constraint_mask: tuple[bool, ...]
    ordering: MagneticOrdering = MagneticOrdering.NONMAGNETIC
    propagation_vector: tuple[float, float, float] | None = None
    orbit_phases: tuple[float, ...] | None = None
    global_spin_inversion_equivalent: bool = True
    schema_version: str = MAGNETIC_STATE_IDENTITY_SCHEMA
    magnetic_state_id: str = field(init=False)

    def __post_init__(self) -> None:
        normalized_moments = _normalise_moments(
            self.moments,
            self.constraint_mask,
            atom_order=None,
        )
        normalized_moments = _canonicalize_global_inversion(
            normalized_moments,
            enabled=bool(self.global_spin_inversion_equivalent),
        )
        ordering = MagneticOrdering.parse(self.ordering)
        propagation_vector = _normalise_optional_vector(
            self.propagation_vector,
            name="propagation_vector",
        )
        orbit_phases = _normalise_phases(self.orbit_phases)
        constraint_mask = tuple(bool(value) for value in self.constraint_mask)
        object.__setattr__(self, "moments", normalized_moments)
        object.__setattr__(self, "constraint_mask", constraint_mask)
        object.__setattr__(self, "ordering", ordering)
        object.__setattr__(self, "propagation_vector", propagation_vector)
        object.__setattr__(self, "orbit_phases", orbit_phases)
        object.__setattr__(
            self,
            "magnetic_state_id",
            "magnetic_state_" + sha256_canonical_json(self.identity_payload()),
        )

    @classmethod
    def from_components(
        cls,
        moments: Any,
        constraint_mask: Sequence[Any],
        *,
        ordering: MagneticOrdering | str = MagneticOrdering.NONMAGNETIC,
        atom_order: Sequence[int] | None = None,
        canonical_atom_order: Sequence[int] | None = None,
        propagation_vector: Sequence[Any] | None = None,
        orbit_phases: Sequence[Any] | None = None,
        global_spin_inversion_equivalent: bool = True,
    ) -> "MagneticStateIdentity":
        if atom_order is not None and canonical_atom_order is not None:
            raise ValueError("provide only one of atom_order or canonical_atom_order")
        if canonical_atom_order is not None:
            atom_order = canonical_atom_order
        normalized_moments = _normalise_moments(
            moments,
            constraint_mask,
            atom_order=atom_order,
        )
        normalized_mask = tuple(bool(value) for value in constraint_mask)
        if atom_order is not None:
            order = tuple(int(index) for index in atom_order)
            normalized_mask = tuple(normalized_mask[index] for index in order)
        return cls(
            moments=normalized_moments,
            constraint_mask=normalized_mask,
            ordering=MagneticOrdering.parse(ordering),
            propagation_vector=_normalise_optional_vector(
                propagation_vector,
                name="propagation_vector",
            ),
            orbit_phases=_normalise_phases(orbit_phases),
            global_spin_inversion_equivalent=global_spin_inversion_equivalent,
        )

    @property
    def canonical_atom_order(self) -> tuple[int, ...]:
        """Return the canonical positions represented by the stored vectors."""

        return tuple(range(len(self.moments)))

    @classmethod
    def from_moments(
        cls,
        moments: Any,
        constraint_mask: Sequence[Any],
        **kwargs: Any,
    ) -> "MagneticStateIdentity":
        return cls.from_components(moments, constraint_mask, **kwargs)

    @classmethod
    def from_atoms(
        cls,
        atoms: Any,
        moments: Any,
        constraint_mask: Sequence[Any],
        **kwargs: Any,
    ) -> "MagneticStateIdentity":
        return cls.from_components(
            moments,
            constraint_mask,
            atom_order=canonical_atom_order(atoms),
            **kwargs,
        )

    def identity_payload(self) -> dict[str, Any]:
        """Return only versioned, state-defining identity fields."""

        return {
            "schema_version": self.schema_version,
            "canonical_atom_order": list(self.canonical_atom_order),
            "ordering": self.ordering.value,
            "moments": [list(vector) for vector in self.moments],
            "constraint_mask": list(self.constraint_mask),
            "global_spin_inversion_equivalent": self.global_spin_inversion_equivalent,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "magnetic_state_id": self.magnetic_state_id,
            **self.identity_payload(),
            "propagation_vector": (
                None if self.propagation_vector is None else list(self.propagation_vector)
            ),
            "orbit_phases": None if self.orbit_phases is None else list(self.orbit_phases),
        }


@dataclass(frozen=True, slots=True)
class MagneticState:
    """Cross-stage magnetic candidate state with its derived identity."""

    ordering: MagneticOrdering
    moment_set_name: str | None
    moments: tuple[tuple[float, float, float], ...]
    constraint_mask: tuple[bool, ...]
    propagation_vector: tuple[float, float, float] | None = None
    orbit_phases: tuple[float, ...] | None = None
    target_net_moment: tuple[float, float, float] | None = None
    compensation: float | None = None
    global_spin_inversion_equivalent: bool = True
    canonical_atom_order: tuple[int, ...] | None = None
    magnetic_state_id: str = field(init=False)

    def __post_init__(self) -> None:
        raw_moments = tuple(_normalise_vector(vector, name="moments") for vector in self.moments)
        order = (
            tuple(range(len(raw_moments)))
            if self.canonical_atom_order is None
            else tuple(int(index) for index in self.canonical_atom_order)
        )
        if sorted(order) != list(range(len(raw_moments))):
            raise ValueError("canonical_atom_order must be a permutation of the moment rows")
        identity = MagneticStateIdentity.from_components(
            raw_moments,
            self.constraint_mask,
            ordering=self.ordering,
            atom_order=order,
            propagation_vector=self.propagation_vector,
            orbit_phases=self.orbit_phases,
            global_spin_inversion_equivalent=self.global_spin_inversion_equivalent,
        )
        # Keep the requested field for transport. Global inversion is an
        # identity-level equivalence, not permission to change the user's
        # deterministic +z FM convention or the generated AFM representative.
        transport_moments = raw_moments
        transport_mask = tuple(bool(value) for value in self.constraint_mask)
        total = tuple(sum(vector[axis] for vector in transport_moments) for axis in range(3))
        target = (
            total
            if self.target_net_moment is None
            else _normalise_vector(self.target_net_moment, name="target_net_moment")
        )
        magnitudes = sum(
            math.sqrt(sum(component * component for component in vector))
            for vector in identity.moments
        )
        compensation = (
            0.0
            if magnitudes == 0.0
            else math.sqrt(sum(component * component for component in total)) / magnitudes
        )
        if self.compensation is not None:
            compensation = float(self.compensation)
            if not math.isfinite(compensation) or compensation < 0.0:
                raise ValueError("compensation must be finite and non-negative")
        object.__setattr__(self, "ordering", identity.ordering)
        object.__setattr__(self, "moments", transport_moments)
        object.__setattr__(self, "constraint_mask", transport_mask)
        object.__setattr__(self, "propagation_vector", identity.propagation_vector)
        object.__setattr__(self, "orbit_phases", identity.orbit_phases)
        object.__setattr__(self, "canonical_atom_order", order)
        object.__setattr__(self, "target_net_moment", target)
        object.__setattr__(self, "compensation", compensation)
        object.__setattr__(self, "magnetic_state_id", identity.magnetic_state_id)

    @property
    def identity(self) -> MagneticStateIdentity:
        return MagneticStateIdentity.from_components(
            moments=self.moments,
            constraint_mask=self.constraint_mask,
            ordering=self.ordering,
            atom_order=self.canonical_atom_order,
            propagation_vector=self.propagation_vector,
            orbit_phases=self.orbit_phases,
            global_spin_inversion_equivalent=self.global_spin_inversion_equivalent,
        )

    def to_extxyz_arrays(self) -> dict[str, list[Any]]:
        """Return JSON/extxyz-safe per-atom arrays."""

        return {
            MAGNETIC_MOMENTS_ARRAY: [list(vector) for vector in self.moments],
            MAGNETIC_CONSTRAINT_MASK_ARRAY: list(self.constraint_mask),
        }

    def to_extxyz_info(self, *, candidate_id: str | None = None) -> dict[str, Any]:
        """Return JSON/extxyz-safe structure metadata."""

        info: dict[str, Any] = {
            "magnetic_ordering": self.ordering.value,
            "magnetic_state_id": self.magnetic_state_id,
            "magnetic_moment_set": self.moment_set_name,
            "magnetic_propagation_vector": (
                None if self.propagation_vector is None else list(self.propagation_vector)
            ),
            "magnetic_orbit_phases": (
                None if self.orbit_phases is None else list(self.orbit_phases)
            ),
            "magnetic_target_net_moment": list(self.target_net_moment or (0.0, 0.0, 0.0)),
            "magnetic_compensation": self.compensation,
            "magnetic_state_schema": MAGNETIC_STATE_IDENTITY_SCHEMA,
        }
        if candidate_id is not None:
            normalized_candidate_id = str(candidate_id).strip()
            if not normalized_candidate_id:
                raise ValueError("candidate_id must not be blank when supplied")
            info["candidate_id"] = normalized_candidate_id
        return info

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity.to_dict(),
            "moment_set_name": self.moment_set_name,
            "target_net_moment": list(self.target_net_moment or (0.0, 0.0, 0.0)),
            "compensation": self.compensation,
        }


# Concise alias for callers that use the terminology from the issue text.
MomentSet = MagneticMomentSet


__all__ = [
    "MAGNETIC_CONSTRAINT_MASK_ARRAY",
    "MAGNETIC_MOMENTS_ARRAY",
    "MAGNETIC_STATE_IDENTITY_SCHEMA",
    "calculate_magnetic_state_id",
    "MagneticMomentSet",
    "MagneticOrdering",
    "MagneticState",
    "MagneticStateIdentity",
    "MomentSet",
    "canonical_atom_order",
]
